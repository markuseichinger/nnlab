//! Datensätze: MNIST und Fashion-MNIST direkt (IDX-Format), dazu beliebige
//! Bildklassifikations-Datensätze von Hugging Face (Parquet). Alles wird in ein
//! einheitliches Format gebracht: quadratische Bilder, 1 oder 3 Kanäle,
//! planar gespeichert (C, H, W) als u8, Labels als u8.

use crate::rng::Rng;
use flate2::read::GzDecoder;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Beschreibung eines Datensatzes
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    Mnist,
    Fashion,
    /// Hugging-Face-Datensatz, z. B. "uoft-cs/cifar10"
    Hf(String),
}

/// Was geladen wird. Für Hugging Face legen `side` und `rgb` fest, auf welche
/// Größe und Farbtiefe die Bilder gebracht werden.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DataSpec {
    pub source: Source,
    pub side: usize,
    pub rgb: bool,
}

impl Default for DataSpec {
    fn default() -> Self {
        Self::mnist()
    }
}

impl DataSpec {
    pub fn mnist() -> Self {
        Self { source: Source::Mnist, side: 28, rgb: false }
    }
    pub fn fashion() -> Self {
        Self { source: Source::Fashion, side: 28, rgb: false }
    }
    pub fn hf(id: &str, side: usize, rgb: bool) -> Self {
        Self { source: Source::Hf(id.trim().to_string()), side, rgb }
    }
    /// Eingabeform (Kanäle, Höhe, Breite), ohne den Datensatz zu laden.
    pub fn shape(&self) -> (usize, usize, usize) {
        match self.source {
            Source::Mnist | Source::Fashion => (1, 28, 28),
            Source::Hf(_) => (if self.rgb { 3 } else { 1 }, self.side, self.side),
        }
    }
    pub fn label(&self) -> String {
        match &self.source {
            Source::Mnist => "MNIST (Ziffern)".into(),
            Source::Fashion => "Fashion-MNIST (Kleidung)".into(),
            Source::Hf(id) => match HF_SUGGESTIONS.iter().find(|s| s.id == id) {
                Some(s) => s.name.to_string(),
                None => format!("Hugging Face: {id}"),
            },
        }
    }
    /// Grobe Art der Daten, für Berater und Tutorial.
    pub fn kind(&self) -> DataKind {
        match &self.source {
            Source::Mnist => DataKind::Digits,
            Source::Fashion => DataKind::Fashion,
            Source::Hf(id) if id == "ylecun/mnist" => DataKind::Digits,
            Source::Hf(id) if id.contains("kmnist") || id.contains("emnist") => DataKind::Glyphs,
            Source::Hf(id) if id.contains("fashion") => DataKind::Fashion,
            Source::Hf(_) if self.rgb => DataKind::Photos,
            Source::Hf(_) => DataKind::Other,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataKind {
    Digits,
    Glyphs,
    Fashion,
    Photos,
    Other,
}

pub struct HfSuggestion {
    pub id: &'static str,
    pub name: &'static str,
    pub side: usize,
    pub rgb: bool,
    pub info: &'static str,
}

pub const HF_SUGGESTIONS: [HfSuggestion; 5] = [
    HfSuggestion { id: "tanganke/kmnist", name: "KMNIST (japanische Kursivschrift)", side: 28, rgb: false, info: "10 Hiragana-Zeichen, 70 000 Bilder. Format wie MNIST, aber deutlich schwerer." },
    HfSuggestion { id: "tanganke/emnist_letters", name: "EMNIST Buchstaben", side: 28, rgb: false, info: "26 Klassen (A–Z), 145 600 handgeschriebene Buchstaben." },
    HfSuggestion { id: "uoft-cs/cifar10", name: "CIFAR-10 (Farbfotos)", side: 32, rgb: true, info: "10 Klassen, 60 000 Fotos à 32×32 RGB. Der klassische Härtetest: Hier scheitern reine MLPs." },
    HfSuggestion { id: "blanchon/EuroSAT_RGB", name: "EuroSAT (Satellitenbilder)", side: 32, rgb: true, info: "10 Landnutzungsklassen aus Sentinel-2, 27 000 Bilder, auf 32×32 verkleinert." },
    HfSuggestion { id: "ylecun/mnist", name: "MNIST über Hugging Face", side: 28, rgb: false, info: "Dasselbe wie das eingebaute MNIST, zum Ausprobieren des Imports." },
];

pub struct Dataset {
    pub spec: DataSpec,
    pub name: String,
    pub classes: Vec<String>,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub train_x: Vec<u8>,
    pub train_y: Vec<u8>,
    pub test_x: Vec<u8>,
    pub test_y: Vec<u8>,
    /// Mittelwert und Standardabweichung pro Kanal (Pixel / 255), aus dem Trainingsset
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl Dataset {
    pub fn img_len(&self) -> usize {
        self.c * self.h * self.w
    }
    pub fn n_train(&self) -> usize {
        self.train_y.len()
    }
    pub fn n_test(&self) -> usize {
        self.test_y.len()
    }
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }
    pub fn test_image(&self, i: usize) -> &[u8] {
        let n = self.img_len();
        &self.test_x[i * n..][..n]
    }
    pub fn train_image(&self, i: usize) -> &[u8] {
        let n = self.img_len();
        &self.train_x[i * n..][..n]
    }

    /// Prüft die Daten und berechnet Mittelwert/Standardabweichung pro Kanal.
    pub fn finish(mut self) -> Result<Self, String> {
        let n = self.img_len();
        if self.train_x.len() != self.train_y.len() * n || self.test_x.len() != self.test_y.len() * n {
            return Err("Bilder und Labels passen nicht zusammen".into());
        }
        if self.n_train() == 0 || self.n_test() == 0 {
            return Err("Der Datensatz enthält keine Bilder".into());
        }
        let k = self.classes.len();
        if let Some(&bad) = self.train_y.iter().chain(&self.test_y).find(|&&y| y as usize >= k) {
            return Err(format!("Label {bad} liegt außerhalb der {k} Klassen"));
        }
        let hw = self.h * self.w;
        let (mut mean, mut std) = (vec![0.0f32; self.c], vec![0.0f32; self.c]);
        let sample = self.n_train().min(10_000);
        for ch in 0..self.c {
            let (mut s, mut s2, mut cnt) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..sample {
                for &p in &self.train_x[i * n + ch * hw..][..hw] {
                    let v = p as f64 / 255.0;
                    s += v;
                    s2 += v * v;
                    cnt += 1.0;
                }
            }
            let m = s / cnt;
            mean[ch] = m as f32;
            std[ch] = ((s2 / cnt - m * m).max(1e-6)).sqrt() as f32;
        }
        self.mean = mean;
        self.std = std;
        Ok(self)
    }
}

pub fn cache_dir() -> PathBuf {
    dirs::data_dir().unwrap_or_else(|| PathBuf::from(".")).join("nnlab")
}

/// Lädt den Datensatz aus dem Cache oder aus dem Netz.
pub fn load(spec: &DataSpec, progress: &dyn Fn(String)) -> Result<Dataset, String> {
    match &spec.source {
        Source::Mnist | Source::Fashion => load_idx(spec, progress),
        Source::Hf(id) => load_hf(spec, id, progress),
    }
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().timeout_global(Some(std::time::Duration::from_secs(900))).user_agent("nnlab/0.2").build().into()
}

/// Lädt eine URL komplett, mit Fortschrittsanzeige in MB.
fn fetch(url: &str, what: &str, progress: &dyn Fn(String)) -> Result<Vec<u8>, String> {
    let mut r = agent().get(url).call().map_err(|e| format!("{url}: {e}"))?;
    let total = r.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<u64>().ok());
    let mut body = r.body_mut().with_config().limit(4 << 30).reader();
    let mut out = Vec::with_capacity(total.unwrap_or(0).min(1 << 30) as usize);
    let mut buf = vec![0u8; 1 << 16];
    let mut last = std::time::Instant::now();
    loop {
        let n = body.read(&mut buf).map_err(|e| format!("{url}: {e}"))?;
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n]);
        if last.elapsed().as_millis() > 250 {
            last = std::time::Instant::now();
            let mb = out.len() as f64 / 1e6;
            progress(match total {
                Some(t) => format!("Lade {what}: {mb:.1} von {:.1} MB", t as f64 / 1e6),
                None => format!("Lade {what}: {mb:.1} MB"),
            });
        }
    }
    Ok(out)
}

fn write_atomic(path: &Path, d: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension("part");
    std::fs::write(&tmp, d).and_then(|_| std::fs::rename(&tmp, path)).map_err(|e| format!("Speichern {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// MNIST / Fashion-MNIST im IDX-Format
// ---------------------------------------------------------------------------

const IDX_FILES: [&str; 4] = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"];

fn load_idx(spec: &DataSpec, progress: &dyn Fn(String)) -> Result<Dataset, String> {
    let (dir, mirrors, classes): (&str, &[&str], Vec<String>) = match spec.source {
        Source::Mnist => (
            "mnist",
            &["https://storage.googleapis.com/cvdf-datasets/mnist/", "https://ossci-datasets.s3.amazonaws.com/mnist/", "https://raw.githubusercontent.com/fgnt/mnist/master/"],
            (0..10).map(|i| i.to_string()).collect(),
        ),
        _ => (
            "fashion-mnist",
            &["https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/", "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"],
            ["T-Shirt", "Hose", "Pullover", "Kleid", "Mantel", "Sandale", "Hemd", "Sneaker", "Tasche", "Stiefel"].iter().map(|s| s.to_string()).collect(),
        ),
    };
    let dir = cache_dir().join(dir);
    std::fs::create_dir_all(&dir).map_err(|e| format!("Cache-Ordner {}: {e}", dir.display()))?;
    let mut raw = Vec::new();
    for f in IDX_FILES {
        let path = dir.join(f);
        let gz = match std::fs::read(&path) {
            Ok(d) if !d.is_empty() => d,
            _ => {
                let mut errs = Vec::new();
                let mut got = None;
                for m in mirrors {
                    match fetch(&format!("{m}{f}"), f, progress) {
                        Ok(d) if d.len() > 2 && d[0] == 0x1f && d[1] == 0x8b => {
                            got = Some(d);
                            break;
                        }
                        Ok(_) => errs.push(format!("{m}{f}: keine gzip-Datei")),
                        Err(e) => errs.push(e),
                    }
                }
                let d = got.ok_or_else(|| format!("Download fehlgeschlagen:\n{}\n\nAlternativ die vier .gz-Dateien manuell nach\n{}\nlegen.", errs.join("\n"), dir.display()))?;
                write_atomic(&path, &d)?;
                d
            }
        };
        let mut out = Vec::new();
        GzDecoder::new(&gz[..]).read_to_end(&mut out).map_err(|e| {
            let _ = std::fs::remove_file(&path);
            format!("{f} ist beschädigt ({e}); bitte erneut versuchen")
        })?;
        raw.push(out);
    }
    progress("Daten werden eingelesen …".into());
    Dataset {
        spec: spec.clone(),
        name: spec.label(),
        classes,
        c: 1,
        h: 28,
        w: 28,
        train_x: idx(&raw[0], 3, 0x0803)?,
        train_y: idx(&raw[1], 1, 0x0801)?,
        test_x: idx(&raw[2], 3, 0x0803)?,
        test_y: idx(&raw[3], 1, 0x0801)?,
        mean: Vec::new(),
        std: Vec::new(),
    }
    .finish()
}

/// IDX-Format: Magic, Dimensionen (big endian), dann die Bytes.
fn idx(d: &[u8], dims: usize, magic: u32) -> Result<Vec<u8>, String> {
    let be = |i: usize| u32::from_be_bytes([d[i], d[i + 1], d[i + 2], d[i + 3]]);
    if d.len() < 4 + 4 * dims || be(0) != magic {
        return Err("Unerwartetes Dateiformat (IDX)".into());
    }
    let n: usize = (0..dims).map(|k| be(4 + 4 * k) as usize).product();
    let start = 4 + 4 * dims;
    if d.len() < start + n {
        return Err("Datei ist unvollständig".into());
    }
    Ok(d[start..start + n].to_vec())
}

// ---------------------------------------------------------------------------
// Hugging Face (Parquet)
// ---------------------------------------------------------------------------

const MAX_TRAIN: usize = 150_000;
const MAX_TEST: usize = 30_000;

/// Welche Spalten Bild und Label enthalten und wie die Klassen heißen.
#[derive(Debug, Clone)]
pub struct HfLayout {
    pub config: String,
    pub image_col: String,
    pub label_col: String,
    pub classes: Vec<String>,
}

fn hf_cache_file(spec: &DataSpec, id: &str) -> PathBuf {
    let safe: String = id.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '.' { c } else { '_' }).collect();
    cache_dir().join("hf").join(safe).join(format!("{}px_{}.nnlab", spec.side, if spec.rgb { "rgb" } else { "gray" }))
}

fn load_hf(spec: &DataSpec, id: &str, progress: &dyn Fn(String)) -> Result<Dataset, String> {
    if id.is_empty() || !id.contains('/') {
        return Err("Bitte eine Hugging-Face-ID der Form „besitzer/name“ angeben, z. B. uoft-cs/cifar10".into());
    }
    if !(8..=128).contains(&spec.side) {
        return Err("Die Bildgröße muss zwischen 8 und 128 Pixeln liegen".into());
    }
    let cache = hf_cache_file(spec, id);
    if let Ok(d) = std::fs::read(&cache) {
        progress("Lese Cache …".into());
        if let Ok(ds) = read_cache(&d, spec) {
            return Ok(ds);
        }
    }
    progress(format!("Frage {id} bei Hugging Face ab …"));
    let info = fetch(&format!("https://datasets-server.huggingface.co/info?dataset={id}"), "Datensatz-Info", progress)?;
    let info: serde_json::Value = serde_json::from_slice(&info).map_err(|e| format!("Antwort von Hugging Face unlesbar: {e}"))?;
    if let Some(e) = info.get("error").and_then(|e| e.as_str()) {
        return Err(format!("Hugging Face meldet: {e}"));
    }
    let layout = parse_info(&info)?;
    let files = fetch(&format!("https://huggingface.co/api/datasets/{id}/parquet"), "Dateiliste", progress)?;
    let files: serde_json::Value = serde_json::from_slice(&files).map_err(|e| format!("Dateiliste unlesbar: {e}"))?;
    let (train_urls, test_urls) = parquet_urls(&files, &layout.config)?;

    let mut train = Collect::new(spec, MAX_TRAIN);
    for (i, u) in train_urls.iter().enumerate() {
        if train.full() {
            break;
        }
        let bytes = fetch(u, &format!("Trainingsdaten, Teil {}/{}", i + 1, train_urls.len()), progress)?;
        progress(format!("Dekodiere Trainingsbilder ({} bisher) …", train.y.len()));
        read_parquet(bytes, &layout, &mut train)?;
    }
    let mut test = Collect::new(spec, MAX_TEST);
    for (i, u) in test_urls.iter().enumerate() {
        if test.full() {
            break;
        }
        let bytes = fetch(u, &format!("Testdaten, Teil {}/{}", i + 1, test_urls.len()), progress)?;
        progress("Dekodiere Testbilder …".into());
        read_parquet(bytes, &layout, &mut test)?;
    }
    let ds = assemble(spec, &layout, train, test)?;
    if let Some(dir) = cache.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = write_atomic(&cache, &write_cache(&ds));
    Ok(ds)
}

/// Train- und Test-URLs aus der Parquet-Dateiliste (Test notfalls „validation“).
pub fn parquet_urls(files: &serde_json::Value, config: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let splits = files.get(config).and_then(|v| v.as_object()).ok_or_else(|| format!("Keine Parquet-Dateien für die Konfiguration „{config}“ gefunden"))?;
    let urls = |name: &str| -> Vec<String> { splits.get(name).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|u| u.as_str().map(String::from)).collect()).unwrap_or_default() };
    let train = urls("train");
    if train.is_empty() {
        return Err(format!("Der Datensatz hat keinen „train“-Split (vorhanden: {})", splits.keys().cloned().collect::<Vec<_>>().join(", ")));
    }
    let test = if !urls("test").is_empty() { urls("test") } else { urls("validation") };
    Ok((train, test))
}

/// Setzt die gesammelten Bilder zum Datensatz zusammen; ohne Test-Split
/// werden 15 % des Trainings abgezweigt.
fn assemble(spec: &DataSpec, layout: &HfLayout, mut train: Collect, mut test: Collect) -> Result<Dataset, String> {
    let (c, h, w) = spec.shape();
    if test.y.is_empty() {
        let n = train.y.len();
        let k = (n * 15 / 100).max(1).min(n.saturating_sub(1));
        test.x = train.x.split_off((n - k) * c * h * w);
        test.y = train.y.split_off(n - k);
    }
    Dataset {
        spec: spec.clone(),
        name: spec.label(),
        classes: layout.classes.iter().map(|s| displayable(s)).collect(),
        c,
        h,
        w,
        train_x: train.x,
        train_y: train.y,
        test_x: test.x,
        test_y: test.y,
        mean: Vec::new(),
        std: Vec::new(),
    }
    .finish()
}

/// Sucht in der Datensatz-Info eine Konfiguration mit Bild- und Klassenlabel-Spalte.
pub fn parse_info(info: &serde_json::Value) -> Result<HfLayout, String> {
    let di = info.get("dataset_info").and_then(|v| v.as_object()).ok_or("Die Antwort enthält keine „dataset_info“")?;
    let mut found_other = Vec::new();
    for (config, ci) in di {
        let Some(feats) = ci.get("features").and_then(|v| v.as_object()) else { continue };
        let ty = |v: &serde_json::Value| v.get("_type").and_then(|t| t.as_str()).unwrap_or("").to_string();
        let image = feats.iter().find(|(_, v)| ty(v) == "Image").map(|(k, _)| k.clone());
        let label = feats.iter().find(|(_, v)| ty(v) == "ClassLabel").map(|(k, v)| {
            let names = v.get("names").and_then(|n| n.as_array()).map(|a| a.iter().map(|s| s.as_str().unwrap_or("?").to_string()).collect::<Vec<_>>()).unwrap_or_default();
            (k.clone(), names)
        });
        match (image, label) {
            (Some(img), Some((lab, names))) if !names.is_empty() => {
                if names.len() > 256 {
                    return Err(format!("{} Klassen werden nicht unterstützt (höchstens 256)", names.len()));
                }
                return Ok(HfLayout { config: config.clone(), image_col: img, label_col: lab, classes: names });
            }
            _ => found_other.push(format!("{config}: {}", feats.iter().map(|(k, v)| format!("{k} ({})", ty(v))).collect::<Vec<_>>().join(", "))),
        }
    }
    Err(format!("Kein Bildklassifikations-Datensatz: Es braucht eine Spalte vom Typ Image und eine vom Typ ClassLabel.\nGefunden:\n{}", found_other.join("\n")))
}

/// Klassennamen in Schriften, die die Oberfläche nicht darstellen kann
/// (z. B. Japanisch), werden umschrieben.
fn displayable(s: &str) -> String {
    const KANA: [(&str, &str); 10] = [("お", "o"), ("き", "ki"), ("す", "su"), ("つ", "tsu"), ("な", "na"), ("は", "ha"), ("ま", "ma"), ("や", "ya"), ("れ", "re"), ("を", "wo")];
    if let Some((_, r)) = KANA.iter().find(|(k, _)| *k == s) {
        return r.to_string();
    }
    if s.chars().any(|c| !(c.is_ascii() || ('\u{a0}'..='\u{24f}').contains(&c))) {
        let ascii: String = s.chars().filter(|c| c.is_ascii()).collect();
        return if ascii.trim().is_empty() { format!("#{}", s.len()) } else { ascii.trim().to_string() };
    }
    s.to_string()
}

struct Collect {
    side: usize,
    rgb: bool,
    max: usize,
    x: Vec<u8>,
    y: Vec<u8>,
}

impl Collect {
    fn new(spec: &DataSpec, max: usize) -> Self {
        Self { side: spec.side, rgb: spec.rgb, max, x: Vec::new(), y: Vec::new() }
    }
    fn full(&self) -> bool {
        self.y.len() >= self.max
    }
    /// Dekodiert ein Bild, bringt es auf side×side und legt es planar ab.
    fn push(&mut self, bytes: &[u8], label: i64) -> Result<(), String> {
        if !(0..=255).contains(&label) {
            return Err(format!("Label {label} ungültig"));
        }
        let img = image::load_from_memory(bytes).map_err(|e| format!("Bild nicht lesbar: {e}"))?;
        let s = self.side as u32;
        let img = if img.width() == s && img.height() == s { img } else { img.resize_exact(s, s, image::imageops::FilterType::Triangle) };
        let n = self.side * self.side;
        if self.rgb {
            let rgb = img.to_rgb8();
            let raw = rgb.as_raw();
            let base = self.x.len();
            self.x.resize(base + 3 * n, 0);
            for i in 0..n {
                for ch in 0..3 {
                    self.x[base + ch * n + i] = raw[i * 3 + ch];
                }
            }
        } else {
            self.x.extend_from_slice(img.to_luma8().as_raw());
        }
        self.y.push(label as u8);
        Ok(())
    }
}

/// Liest eine lokale Parquet-Datei im Hugging-Face-Format (für Tests).
pub fn dataset_from_parquet(spec: &DataSpec, layout: &HfLayout, train: &Path, test: Option<&Path>) -> Result<Dataset, String> {
    let mut tr = Collect::new(spec, MAX_TRAIN);
    read_parquet(std::fs::read(train).map_err(|e| e.to_string())?, layout, &mut tr)?;
    let mut te = Collect::new(spec, MAX_TEST);
    if let Some(t) = test {
        read_parquet(std::fs::read(t).map_err(|e| e.to_string())?, layout, &mut te)?;
    }
    assemble(spec, layout, tr, te)
}

/// Liest Bild- und Labelspalte (Bild = Struktur {bytes, path}, Label = Ganzzahl).
fn read_parquet(bytes: Vec<u8>, layout: &HfLayout, out: &mut Collect) -> Result<(), String> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::Field;
    let reader = SerializedFileReader::new(bytes::Bytes::from(bytes)).map_err(|e| format!("Parquet-Datei unlesbar: {e}"))?;
    let rows = reader.get_row_iter(None).map_err(|e| format!("Parquet: {e}"))?;
    for row in rows {
        if out.full() {
            break;
        }
        let row = row.map_err(|e| format!("Parquet: {e}"))?;
        let (mut img, mut label) = (None, None);
        for (name, f) in row.get_column_iter() {
            if *name == layout.image_col {
                img = match f {
                    Field::Group(g) => g.get_column_iter().find_map(|(k, v)| match v {
                        Field::Bytes(b) if k == "bytes" => Some(b.data().to_vec()),
                        _ => None,
                    }),
                    Field::Bytes(b) => Some(b.data().to_vec()),
                    _ => None,
                };
            } else if *name == layout.label_col {
                label = match f {
                    Field::Byte(v) => Some(*v as i64),
                    Field::Short(v) => Some(*v as i64),
                    Field::Int(v) => Some(*v as i64),
                    Field::Long(v) => Some(*v),
                    Field::UByte(v) => Some(*v as i64),
                    Field::UShort(v) => Some(*v as i64),
                    Field::UInt(v) => Some(*v as i64),
                    Field::ULong(v) => Some(*v as i64),
                    _ => None,
                };
            }
        }
        match (img, label) {
            (Some(b), Some(l)) if l >= 0 => out.push(&b, l)?,
            (None, _) => return Err(format!("Die Spalte „{}“ enthält keine eingebetteten Bilddaten", layout.image_col)),
            _ => {} // Zeilen ohne Label (z. B. −1) überspringen
        }
    }
    Ok(())
}

// Eigenes, einfaches Cache-Format, damit nicht jedes Mal neu dekodiert wird.
const MAGIC: &[u8; 8] = b"NNLAB01\n";

fn write_cache(ds: &Dataset) -> Vec<u8> {
    let mut o = Vec::with_capacity(ds.train_x.len() + ds.test_x.len() + 1024);
    o.extend_from_slice(MAGIC);
    for v in [ds.c, ds.h, ds.w, ds.classes.len(), ds.n_train(), ds.n_test()] {
        o.extend_from_slice(&(v as u32).to_le_bytes());
    }
    for c in &ds.classes {
        o.extend_from_slice(&(c.len() as u32).to_le_bytes());
        o.extend_from_slice(c.as_bytes());
    }
    for part in [&ds.train_x, &ds.train_y, &ds.test_x, &ds.test_y] {
        o.write_all(part).unwrap();
    }
    o
}

struct Cursor<'a> {
    d: &'a [u8],
    p: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let v = self.d.get(self.p..self.p + n).ok_or("Cache unvollständig")?;
        self.p += n;
        Ok(v)
    }
    fn u32(&mut self) -> Result<usize, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }
}

fn read_cache(d: &[u8], spec: &DataSpec) -> Result<Dataset, String> {
    if d.len() < 32 || &d[..8] != MAGIC {
        return Err("kein Cache".into());
    }
    let mut cur = Cursor { d, p: 8 };
    let (c, h, w, k, ntr, nte) = (cur.u32()?, cur.u32()?, cur.u32()?, cur.u32()?, cur.u32()?, cur.u32()?);
    let mut classes = Vec::new();
    for _ in 0..k {
        let l = cur.u32()?;
        classes.push(String::from_utf8_lossy(cur.take(l)?).into_owned());
    }
    let il = c * h * w;
    let train_x = cur.take(ntr * il)?.to_vec();
    let train_y = cur.take(ntr)?.to_vec();
    let test_x = cur.take(nte * il)?.to_vec();
    let test_y = cur.take(nte)?.to_vec();
    Dataset { spec: spec.clone(), name: spec.label(), classes, c, h, w, train_x, train_y, test_x, test_y, mean: Vec::new(), std: Vec::new() }.finish()
}

// ---------------------------------------------------------------------------
// Vorverarbeitung
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    /// Pixel / 255 → [0, 1]
    Unit,
    /// (Pixel/255 − μ) / σ pro Kanal, aus dem Trainingsset
    Standard,
    /// Pixel / 255 · 2 − 1 → [−1, 1]
    Symmetric,
}

impl Norm {
    pub const ALL: [Norm; 3] = [Norm::Unit, Norm::Standard, Norm::Symmetric];
    pub fn name(self) -> &'static str {
        match self {
            Norm::Unit => "[0, 1]",
            Norm::Standard => "Standardisiert (μ=0, σ=1)",
            Norm::Symmetric => "[−1, 1]",
        }
    }
    #[inline]
    pub fn apply(self, ds: &Dataset, ch: usize, p: u8) -> f32 {
        let v = p as f32 / 255.0;
        match self {
            Norm::Unit => v,
            Norm::Symmetric => v * 2.0 - 1.0,
            Norm::Standard => (v - ds.mean[ch]) / ds.std[ch],
        }
    }
}

/// Augmentierung beim Training.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Augment {
    /// Zufällige Verschiebung um bis zu ±shift Pixel
    pub shift: usize,
    /// Zufälliges horizontales Spiegeln (für Fotos, nicht für Ziffern oder Buchstaben)
    pub flip: bool,
}

/// Schreibt ein Bild normalisiert in `out`; mit `rng` wird augmentiert.
pub fn prepare(img: &[u8], out: &mut [f32], ds: &Dataset, norm: Norm, aug: Augment, rng: Option<&mut Rng>) {
    let (h, w) = (ds.h as isize, ds.w as isize);
    let (dx, dy, flip) = match rng {
        Some(r) => {
            let s = aug.shift;
            let (dx, dy) = if s > 0 { (r.below(2 * s + 1) as isize - s as isize, r.below(2 * s + 1) as isize - s as isize) } else { (0, 0) };
            (dx, dy, aug.flip && r.below(2) == 1)
        }
        None => (0, 0, false),
    };
    let hw = (h * w) as usize;
    for ch in 0..ds.c {
        let bg = norm.apply(ds, ch, 0);
        let src = &img[ch * hw..][..hw];
        let dst = &mut out[ch * hw..][..hw];
        for y in 0..h {
            for x in 0..w {
                let (mut sx, sy) = (x - dx, y - dy);
                if flip {
                    sx = w - 1 - sx;
                }
                dst[(y * w + x) as usize] = if sx < 0 || sy < 0 || sx >= w || sy >= h { bg } else { norm.apply(ds, ch, src[(sy * w + sx) as usize]) };
            }
        }
    }
}
