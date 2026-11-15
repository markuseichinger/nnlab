//! MNIST und Fashion-MNIST: Download (mehrere Mirrors), Cache, IDX-Format.

use crate::rng::Rng;
use flate2::read::GzDecoder;
use std::io::Read;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetKind {
    Mnist,
    Fashion,
}

impl DatasetKind {
    pub fn name(self) -> &'static str {
        match self {
            DatasetKind::Mnist => "MNIST (Ziffern)",
            DatasetKind::Fashion => "Fashion-MNIST (Kleidung)",
        }
    }
    fn dir(self) -> &'static str {
        match self {
            DatasetKind::Mnist => "mnist",
            DatasetKind::Fashion => "fashion-mnist",
        }
    }
    fn mirrors(self) -> &'static [&'static str] {
        match self {
            DatasetKind::Mnist => &[
                "https://storage.googleapis.com/cvdf-datasets/mnist/",
                "https://ossci-datasets.s3.amazonaws.com/mnist/",
                "https://raw.githubusercontent.com/fgnt/mnist/master/",
            ],
            DatasetKind::Fashion => &[
                "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
            ],
        }
    }
    pub fn class_names(self) -> [&'static str; 10] {
        match self {
            DatasetKind::Mnist => ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            DatasetKind::Fashion => ["T-Shirt", "Hose", "Pullover", "Kleid", "Mantel", "Sandale", "Hemd", "Sneaker", "Tasche", "Stiefel"],
        }
    }
}

pub const SIDE: usize = 28;
pub const PIX: usize = SIDE * SIDE;

pub struct Dataset {
    pub kind: DatasetKind,
    pub train_x: Vec<u8>,
    pub train_y: Vec<u8>,
    pub test_x: Vec<u8>,
    pub test_y: Vec<u8>,
}

impl Dataset {
    pub fn n_train(&self) -> usize {
        self.train_y.len()
    }
    pub fn n_test(&self) -> usize {
        self.test_y.len()
    }
    pub fn test_image(&self, i: usize) -> &[u8] {
        &self.test_x[i * PIX..][..PIX]
    }
    pub fn train_image(&self, i: usize) -> &[u8] {
        &self.train_x[i * PIX..][..PIX]
    }
}

pub fn cache_dir() -> PathBuf {
    dirs::data_dir().unwrap_or_else(|| PathBuf::from(".")).join("nnlab")
}

const FILES: [&str; 4] = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"];

/// Lädt den Datensatz, aus dem Cache oder aus dem Netz.
pub fn load(kind: DatasetKind, progress: &dyn Fn(String)) -> Result<Dataset, String> {
    let dir = cache_dir().join(kind.dir());
    std::fs::create_dir_all(&dir).map_err(|e| format!("Cache-Ordner {}: {e}", dir.display()))?;
    let mut raw = Vec::new();
    for f in FILES {
        let path = dir.join(f);
        let gz = match std::fs::read(&path) {
            Ok(d) if !d.is_empty() => d,
            _ => {
                let d = download(kind, f, progress)?;
                let tmp = path.with_extension("part");
                std::fs::write(&tmp, &d).and_then(|_| std::fs::rename(&tmp, &path)).map_err(|e| format!("Speichern {}: {e}", path.display()))?;
                d
            }
        };
        let mut out = Vec::new();
        GzDecoder::new(&gz[..]).read_to_end(&mut out).map_err(|e| {
            let _ = std::fs::remove_file(&path);
            format!("{f} ist beschädigt ({e}); bitte erneut starten")
        })?;
        raw.push(out);
    }
    progress("Daten werden eingelesen …".into());
    let train_x = idx(&raw[0], 3, 0x0803)?;
    let train_y = idx(&raw[1], 1, 0x0801)?;
    let test_x = idx(&raw[2], 3, 0x0803)?;
    let test_y = idx(&raw[3], 1, 0x0801)?;
    if train_x.len() != train_y.len() * PIX || test_x.len() != test_y.len() * PIX {
        return Err("Bild- und Labeldateien passen nicht zusammen".into());
    }
    Ok(Dataset { kind, train_x, train_y, test_x, test_y })
}

fn download(kind: DatasetKind, file: &str, progress: &dyn Fn(String)) -> Result<Vec<u8>, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(std::time::Duration::from_secs(120))).user_agent("nnlab/0.1").build().into();
    let mut errs = Vec::new();
    for m in kind.mirrors() {
        let url = format!("{m}{file}");
        progress(format!("Lade {file} …"));
        match agent.get(&url).call() {
            Ok(mut r) => match r.body_mut().with_config().limit(64 << 20).read_to_vec() {
                Ok(d) if d.len() > 2 && d[0] == 0x1f && d[1] == 0x8b => return Ok(d),
                Ok(_) => errs.push(format!("{url}: keine gzip-Datei")),
                Err(e) => errs.push(format!("{url}: {e}")),
            },
            Err(e) => errs.push(format!("{url}: {e}")),
        }
    }
    Err(format!(
        "Download fehlgeschlagen:\n{}\n\nAlternativ die vier .gz-Dateien manuell nach\n{}\nlegen.",
        errs.join("\n"),
        cache_dir().join(kind.dir()).display()
    ))
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
// Vorverarbeitung
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    /// Pixel / 255 → [0, 1]
    Unit,
    /// (Pixel/255 − μ) / σ mit den Werten des Trainingssets
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
    pub fn apply(self, kind: DatasetKind, p: u8) -> f32 {
        let v = p as f32 / 255.0;
        match self {
            Norm::Unit => v,
            Norm::Symmetric => v * 2.0 - 1.0,
            Norm::Standard => {
                let (mu, sd) = match kind {
                    DatasetKind::Mnist => (0.1307, 0.3081),
                    DatasetKind::Fashion => (0.2860, 0.3530),
                };
                (v - mu) / sd
            }
        }
    }
}

/// Schreibt ein Bild normalisiert in `out`; optional zufällig um ±`shift` Pixel verschoben.
pub fn prepare(img: &[u8], out: &mut [f32], norm: Norm, kind: DatasetKind, shift: usize, rng: Option<&mut Rng>) {
    let (dx, dy) = match rng {
        Some(r) if shift > 0 => (r.below(2 * shift + 1) as isize - shift as isize, r.below(2 * shift + 1) as isize - shift as isize),
        _ => (0, 0),
    };
    let bg = norm.apply(kind, 0);
    for y in 0..SIDE as isize {
        for x in 0..SIDE as isize {
            let (sx, sy) = (x - dx, y - dy);
            out[(y * SIDE as isize + x) as usize] =
                if sx < 0 || sy < 0 || sx >= SIDE as isize || sy >= SIDE as isize { bg } else { norm.apply(kind, img[(sy * SIDE as isize + sx) as usize]) };
        }
    }
}
