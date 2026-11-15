//! nnlab: neuronale Netze konfigurieren, trainieren und beim Lernen zusehen.

mod draw;
mod viz;

use eframe::egui::{self, Color32, ColorImage, RichText, Sense, TextureHandle, TextureOptions};
use egui_plot::{Bar, BarChart, HLine, Legend, Line, Plot, PlotPoints, Points};
use nnlab::data::{self, Dataset, DatasetKind, Norm, PIX, SIDE};
use nnlab::nn::{self, ActKind, Init, Layer, LayerSpec, LossKind, Model, OptKind};
use nnlab::trainer::{self, Schedule, Status, TrainConfig, Trainer};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn main() -> eframe::Result {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("nnlab – Neuronale Netze trainieren").with_inner_size([1560.0, 980.0]).with_min_inner_size([1100.0, 700.0]),
        ..Default::default()
    };
    eframe::run_native("nnlab", opts, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

const ACCENT: Color32 = Color32::from_rgb(255, 150, 50);
const BLUE: Color32 = Color32::from_rgb(70, 160, 255);
const GREEN: Color32 = Color32::from_rgb(120, 209, 81);
const RED: Color32 = Color32::from_rgb(240, 90, 80);
const MUTED: Color32 = Color32::from_rgb(140, 145, 155);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Verlauf,
    Netz,
    Gewichte,
    Dynamik,
    Auswertung,
    Zeichnen,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Probe {
    Test(usize),
    Drawing,
}

/// Ergebnis eines Vorwärtslaufs für das betrachtete Sample.
struct Trace {
    key: u64,
    img: Vec<u8>,
    input: Vec<f32>,
    outs: Vec<Vec<f32>>,
    probs: Vec<f32>,
}

#[derive(Default)]
struct Textures(HashMap<String, (u64, TextureHandle)>);

impl Textures {
    fn get(&mut self, ctx: &egui::Context, key: &str, ver: u64, make: impl FnOnce() -> ColorImage) -> TextureHandle {
        if let Some((v, t)) = self.0.get(key) {
            if *v == ver {
                return t.clone();
            }
        }
        let img = make();
        if let Some((v, t)) = self.0.get_mut(key) {
            t.set(img, TextureOptions::NEAREST);
            *v = ver;
            return t.clone();
        }
        let t = ctx.load_texture(key, img, TextureOptions::NEAREST);
        self.0.insert(key.to_string(), (ver, t.clone()));
        t
    }
}

fn image(ui: &mut egui::Ui, t: &TextureHandle, w: f32, h: f32) -> egui::Response {
    ui.add(egui::Image::new(egui::load::SizedTexture::new(t.id(), egui::vec2(w, h))))
}

struct Preset {
    name: &'static str,
    info: &'static str,
    layers: Vec<LayerSpec>,
    opt: Option<(OptKind, f32, Init)>,
}

fn presets() -> Vec<Preset> {
    use ActKind::*;
    use LayerSpec::*;
    vec![
        Preset { name: "Softmax-Regression", info: "Keine verdeckte Schicht: ein lineares Modell, ca. 92 %.", layers: vec![], opt: None },
        Preset { name: "MLP 128 ReLU", info: "Der Klassiker: eine verdeckte Schicht, ca. 98 %.", layers: vec![Dense { units: 128 }, Act(ReLU)], opt: None },
        Preset {
            name: "MLP 2×256 + Dropout",
            info: "Breiter und tiefer, mit Dropout gegen Overfitting.",
            layers: vec![Dense { units: 256 }, Act(ReLU), Dropout { p: 0.2 }, Dense { units: 256 }, Act(ReLU), Dropout { p: 0.2 }],
            opt: None,
        },
        Preset {
            name: "Tiefes Sigmoid-MLP (Vanishing Gradients)",
            info: "6× Sigmoid mit SGD: Im Tab „Gradienten-Dynamik“ werden die Gradienten zur Eingabe hin immer kleiner. Danach ReLU oder BatchNorm einsetzen und vergleichen.",
            layers: [Dense { units: 64 }, Act(Sigmoid)].repeat(6),
            opt: Some((OptKind::Sgd, 0.1, Init::Xavier)),
        },
        Preset {
            name: "CNN (LeNet-artig)",
            info: "Zwei Faltungsblöcke mit Pooling, dann Dense. Ca. 99 %.",
            layers: vec![Conv { filters: 8, k: 5, same: true }, Act(ReLU), MaxPool { size: 2 }, Conv { filters: 16, k: 5, same: true }, Act(ReLU), MaxPool { size: 2 }, Dense { units: 64 }, Act(ReLU)],
            opt: None,
        },
        Preset {
            name: "CNN + BatchNorm + Dropout",
            info: "Moderner kleiner CNN. Rechenintensiv: gut geeignet, um die CPU-Last zu sehen.",
            layers: vec![
                Conv { filters: 16, k: 3, same: true },
                BatchNorm,
                Act(ReLU),
                MaxPool { size: 2 },
                Conv { filters: 32, k: 3, same: true },
                BatchNorm,
                Act(ReLU),
                MaxPool { size: 2 },
                Dense { units: 128 },
                BatchNorm,
                Act(ReLU),
                Dropout { p: 0.3 },
            ],
            opt: None,
        },
    ]
}

struct App {
    ctx: egui::Context,
    cfg: TrainConfig,
    tr: Trainer,
    tab: Tab,
    tex: Textures,
    probe: Probe,
    canvas: draw::Canvas,
    smooth: f32,
    log_loss: bool,
    sel_layer: usize,
    preview: Result<Arc<Model>, String>,
    preview_key: Option<(Vec<LayerSpec>, Init, u64)>,
    preview_ver: u64,
    trace: Option<Trace>,
    load_msg: Arc<Mutex<String>>,
    loading: Arc<AtomicBool>,
    lr_factor: f32,
    cores: usize,
    preset_info: Option<&'static str>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let ctx = cc.egui_ctx.clone();
        ctx.set_visuals(egui::Visuals::dark());
        let mut app = Self {
            ctx,
            cfg: TrainConfig::default(),
            tr: Trainer::new(),
            tab: Tab::Verlauf,
            tex: Textures::default(),
            probe: Probe::Test(0),
            canvas: draw::Canvas::default(),
            smooth: 0.9,
            log_loss: true,
            sel_layer: 0,
            preview: Err(String::new()),
            preview_key: None,
            preview_ver: 1 << 40,
            trace: None,
            load_msg: Arc::new(Mutex::new(String::new())),
            loading: Arc::new(AtomicBool::new(false)),
            lr_factor: 1.0,
            cores: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
            preset_info: None,
        };
        app.ensure_data();
        app
    }

    fn dataset(&self) -> Option<Arc<Dataset>> {
        self.tr.data.lock().unwrap().clone()
    }

    /// Lädt den gewählten Datensatz im Hintergrund, falls nötig.
    fn ensure_data(&mut self) {
        if self.dataset().is_some_and(|d| d.kind == self.cfg.dataset) || self.loading.load(Ordering::SeqCst) || self.tr.running() {
            return;
        }
        if self.load_msg.lock().unwrap().starts_with("Fehler") {
            return;
        }
        self.loading.store(true, Ordering::SeqCst);
        let (data, msg, loading, ctx, kind) = (self.tr.data.clone(), self.load_msg.clone(), self.loading.clone(), self.ctx.clone(), self.cfg.dataset);
        std::thread::spawn(move || {
            let r = trainer::ensure_data(kind, &data, &|m| {
                *msg.lock().unwrap() = m;
                ctx.request_repaint();
            });
            *msg.lock().unwrap() = match r {
                Ok(_) => String::new(),
                Err(e) => format!("Fehler: {e}"),
            };
            loading.store(false, Ordering::SeqCst);
            ctx.request_repaint();
        });
    }

    fn update_preview(&mut self) {
        let key = (self.cfg.layers.clone(), self.cfg.init, self.cfg.seed);
        if self.preview_key.as_ref() != Some(&key) {
            self.preview = Model::build(&self.cfg.layers, self.cfg.input_shape(), 10, self.cfg.init, self.cfg.seed).map(Arc::new);
            self.preview_key = Some(key);
            self.preview_ver += 1;
        }
    }

    /// Das angezeigte Modell: das trainierte (bzw. gerade trainierte) oder die untrainierte Vorschau.
    fn model(&self) -> Option<(Arc<Model>, u64, TrainConfig, bool)> {
        let s = self.tr.shared.lock().unwrap();
        if let Some(m) = &s.model {
            return Some((m.clone(), s.model_version, s.cfg.clone(), true));
        }
        drop(s);
        self.preview.as_ref().ok().map(|m| (m.clone(), self.preview_ver, self.cfg.clone(), false))
    }

    fn start(&mut self, paused: bool) {
        let ctx = self.ctx.clone();
        self.lr_factor = 1.0;
        self.tr.start(self.cfg.clone(), paused, move || ctx.request_repaint_after(Duration::from_millis(80)));
    }

    fn reset(&mut self) {
        self.tr.stop();
        let mut s = self.tr.shared.lock().unwrap();
        s.model = None;
        s.hist = Default::default();
        s.last_eval = None;
        s.status = Status::Idle;
        s.epoch = 0;
        s.step = 0;
        s.steps_per_epoch = 0;
        s.samples_per_sec = 0.0;
        s.gflops = 0.0;
        s.elapsed = Duration::ZERO;
    }

    /// Vorwärtslauf für das gewählte Sample (zwischengespeichert).
    fn trace(&mut self) -> Option<(&Trace, Arc<Model>, TrainConfig, bool)> {
        let (model, ver, cfg, trained) = self.model()?;
        let ds = self.dataset();
        if self.probe != Probe::Drawing && ds.is_none() {
            return None;
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (ver, self.probe, cfg.norm as u8, ds.as_ref().map(|d| d.kind as u8)).hash(&mut h);
        if self.probe == Probe::Drawing {
            self.canvas.version.hash(&mut h);
        }
        let key = h.finish();
        if self.trace.as_ref().map(|t| t.key) != Some(key) {
            let img = match self.probe {
                Probe::Test(i) => {
                    let d = ds.as_ref()?;
                    d.test_image(i.min(d.n_test() - 1)).to_vec()
                }
                Probe::Drawing => self.canvas.to_mnist(),
            };
            let mut input = vec![0.0; PIX];
            data::prepare(&img, &mut input, cfg.norm, cfg.dataset, 0, None);
            let mut m = (*model).clone();
            let outs = m.forward_trace(&input);
            let mut probs = vec![0.0; 10];
            nn::softmax(outs.last()?, &mut probs);
            self.trace = Some(Trace { key, img, input, outs, probs });
        }
        Some((self.trace.as_ref()?, model, cfg, trained))
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.update_preview();
        if !self.tr.running() {
            self.ensure_data();
        }
        egui::Panel::top("top").show(ui, |ui| self.top_bar(ui));
        egui::Panel::left("cfg").resizable(true).default_size(390.0).min_size(330.0).show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| self.config_ui(ui));
        });
        egui::CentralPanel::default_margins().show(ui, |ui| {
            ui.horizontal(|ui| {
                for (t, name) in [
                    (Tab::Verlauf, "📈 Verlauf"),
                    (Tab::Netz, "🔍 Netz & Aktivierungen"),
                    (Tab::Gewichte, "▦ Gewichte"),
                    (Tab::Dynamik, "〰 Gradienten-Dynamik"),
                    (Tab::Auswertung, "🎯 Auswertung"),
                    (Tab::Zeichnen, "✏ Zeichnen"),
                ] {
                    ui.selectable_value(&mut self.tab, t, RichText::new(name).size(15.0));
                }
            });
            ui.separator();
            match self.tab {
                Tab::Verlauf => self.tab_verlauf(ui),
                Tab::Netz => self.tab_netz(ui),
                Tab::Gewichte => self.tab_gewichte(ui),
                Tab::Dynamik => self.tab_dynamik(ui),
                Tab::Auswertung => self.tab_auswertung(ui),
                Tab::Zeichnen => self.tab_zeichnen(ui),
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Kopfleiste
// ---------------------------------------------------------------------------

struct TopInfo {
    status: Status,
    epoch: usize,
    step: u64,
    spe: usize,
    sps: f64,
    gflops: f64,
    elapsed: Duration,
    epochs: usize,
    loss: Option<f64>,
    acc: Option<f64>,
    val_acc: Option<f64>,
    n_params: usize,
    macs: usize,
    threads: usize,
    has_model: bool,
}

impl App {
    fn top_bar(&mut self, ui: &mut egui::Ui) {
        let i = {
            let s = self.tr.shared.lock().unwrap();
            let ema = |v: &[[f64; 2]]| viz::curve(&v[v.len().saturating_sub(200)..], 0.95, 1, false).last().map(|p| p[1]);
            TopInfo {
                status: s.status.clone(),
                epoch: s.epoch,
                step: s.step,
                spe: s.steps_per_epoch,
                sps: s.samples_per_sec,
                gflops: s.gflops,
                elapsed: s.elapsed,
                epochs: s.cfg.epochs,
                loss: ema(&s.hist.loss),
                acc: ema(&s.hist.acc),
                val_acc: s.hist.val_acc.last().map(|p| p[1]),
                n_params: s.n_params,
                macs: s.macs,
                threads: s.cfg.threads,
                has_model: s.model.is_some(),
            }
        };
        let running = self.tr.running();
        let paused = i.status == Status::Paused;
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("nnlab").size(20.0).strong().color(ACCENT));
            ui.add_space(10.0);
            let big = |t: &str| egui::Button::new(RichText::new(t).size(15.0));
            if !running {
                let ok = self.preview.is_ok() && !self.loading.load(Ordering::SeqCst);
                if ui.add_enabled(ok, big("▶ Training starten")).on_hover_text("Baut das Netz neu (mit dem Seed) und trainiert.").clicked() {
                    self.start(false);
                }
                if ui.add_enabled(ok, big("⏭ Schrittweise")).on_hover_text("Startet pausiert. Dann trainiert jeder Klick auf „⏭ 1“ genau einen Batch.").clicked() {
                    self.start(true);
                }
                if i.has_model && ui.add(big("↺ Zurücksetzen")).on_hover_text("Trainiertes Modell und Verlauf verwerfen.").clicked() {
                    self.reset();
                }
            } else {
                if paused {
                    if ui.add(big("▶ Fortsetzen")).clicked() {
                        self.tr.ctl.pause.store(false, Ordering::SeqCst);
                    }
                    for n in [1u32, 10, 100] {
                        if ui.add(big(&format!("⏭ {n}"))).on_hover_text(format!("{n} Batch(es) trainieren")).clicked() {
                            self.tr.ctl.steps.fetch_add(n, Ordering::SeqCst);
                        }
                    }
                } else if ui.add(big("⏸ Pause")).clicked() {
                    self.tr.ctl.pause.store(true, Ordering::SeqCst);
                }
                if ui.add(big("⏹ Stopp")).clicked() {
                    self.tr.stop();
                }
                ui.separator();
                ui.label("Lernraten-Faktor live:");
                if ui.add(egui::Slider::new(&mut self.lr_factor, 0.01..=10.0).logarithmic(true).fixed_decimals(2)).on_hover_text("Multipliziert die Lernrate sofort, ohne Neustart.").changed() {
                    self.tr.ctl.lr_scale_milli.store((self.lr_factor * 1000.0) as u32, Ordering::SeqCst);
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let mut d = self.tr.ctl.delay_ms.load(Ordering::Relaxed) as f32;
                if ui
                    .add(egui::Slider::new(&mut d, 0.0..=500.0).text("ms").custom_formatter(|v, _| if v < 1.0 { "aus".into() } else { format!("{v:.0}") }))
                    .on_hover_text("Pause nach jedem Trainingsschritt, um dem Netz beim Lernen zuzusehen. Zählt nicht zur Rechenzeit.")
                    .changed()
                {
                    self.tr.ctl.delay_ms.store(d as u32, Ordering::Relaxed);
                }
                ui.label("🐢 Zeitlupe");
            });
        });
        ui.horizontal_wrapped(|ui| {
            let (txt, col) = match &i.status {
                Status::Idle => ("Bereit".to_string(), MUTED),
                Status::Loading(m) => (m.clone(), BLUE),
                Status::Running => ("Training läuft".into(), GREEN),
                Status::Paused => ("Pausiert".into(), ACCENT),
                Status::Finished => ("Fertig".into(), GREEN),
                Status::Stopped => ("Gestoppt".into(), MUTED),
                Status::Error(e) => (format!("Fehler: {e}"), RED),
            };
            ui.label(RichText::new(txt).color(col).strong());
            let lm = self.load_msg.lock().unwrap().clone();
            if !lm.is_empty() && !running {
                ui.label(RichText::new(lm).color(if self.loading.load(Ordering::SeqCst) { BLUE } else { RED }));
            }
            if i.spe > 0 {
                let in_ep = (i.step as usize).saturating_sub(i.epoch * i.spe).min(i.spe);
                let frac = ((i.epoch as f32 + in_ep as f32 / i.spe as f32) / i.epochs.max(1) as f32).min(1.0);
                ui.add(egui::ProgressBar::new(frac).desired_width(150.0).text(format!("Epoche {}/{}", (i.epoch + 1).min(i.epochs), i.epochs)));
                let m = |ui: &mut egui::Ui, k: &str, v: String| {
                    ui.label(RichText::new(k).color(MUTED));
                    ui.label(RichText::new(v).strong().monospace());
                };
                m(ui, "Schritt", format!("{}", i.step));
                if let Some(l) = i.loss {
                    m(ui, "Verlust", format!("{l:.4}"));
                }
                if let Some(a) = i.acc {
                    m(ui, "Train", format!("{:.2} %", a * 100.0));
                }
                if let Some(a) = i.val_acc {
                    m(ui, "Test", format!("{:.2} %", a * 100.0));
                }
                ui.separator();
                m(ui, "Durchsatz", format!("{:.0} Bilder/s", i.sps));
                m(ui, "Leistung", format!("{:.1} GFLOP/s", i.gflops));
                m(ui, "Threads", format!("{}", i.threads));
                m(ui, "Rechenzeit", format!("{:.1} s", i.elapsed.as_secs_f32()));
                ui.separator();
                m(ui, "Parameter", fmt_n(i.n_params));
                m(ui, "MACs/Bild", fmt_n(i.macs));
            }
        });
        ui.add_space(2.0);
    }
}

fn fmt_n(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2} M", n as f64 / 1e6)
    } else if n >= 10_000 {
        format!("{:.1} k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

// ---------------------------------------------------------------------------
// Konfiguration
// ---------------------------------------------------------------------------

fn combo<T: PartialEq + Copy>(ui: &mut egui::Ui, id: &str, label: &str, v: &mut T, all: &[T], name: impl Fn(T) -> &'static str) -> bool {
    let before = *v;
    ui.horizontal(|ui| {
        ui.label(label);
        egui::ComboBox::from_id_salt(id).selected_text(name(*v)).width(220.0).show_ui(ui, |ui| {
            for &o in all {
                ui.selectable_value(v, o, name(o));
            }
        });
    });
    before != *v
}

impl App {
    fn config_ui(&mut self, ui: &mut egui::Ui) {
        let running = self.tr.running();
        if running {
            ui.label(RichText::new("Während des Trainings gesperrt (bis auf den Lernraten-Faktor oben).").color(MUTED).italics());
        }
        ui.add_enabled_ui(!running, |ui| {
            egui::CollapsingHeader::new(RichText::new("Datensatz").strong()).default_open(true).show(ui, |ui| self.cfg_data(ui));
            egui::CollapsingHeader::new(RichText::new("Architektur").strong()).default_open(true).show(ui, |ui| self.cfg_arch(ui));
            egui::CollapsingHeader::new(RichText::new("Initialisierung & Verlust").strong()).default_open(true).show(ui, |ui| self.cfg_init(ui));
            egui::CollapsingHeader::new(RichText::new("Optimierer").strong()).default_open(true).show(ui, |ui| self.cfg_opt(ui));
            egui::CollapsingHeader::new(RichText::new("Lernraten-Plan").strong()).default_open(true).show(ui, |ui| self.cfg_sched(ui));
            egui::CollapsingHeader::new(RichText::new("Training & Rechenleistung").strong()).default_open(true).show(ui, |ui| self.cfg_train(ui));
        });
    }

    fn cfg_data(&mut self, ui: &mut egui::Ui) {
        let c = &mut self.cfg;
        if combo(ui, "ds", "Daten:", &mut c.dataset, &[DatasetKind::Mnist, DatasetKind::Fashion], |d| d.name()) {
            self.load_msg.lock().unwrap().clear();
        }
        ui.horizontal(|ui| {
            ui.label("Trainingsbeispiele:");
            ui.add(egui::DragValue::new(&mut c.train_size).range(100..=60_000).speed(100));
        })
        .response
        .on_hover_text("Kleine Mengen (z. B. 1000) zeigen Overfitting: Die Train-Genauigkeit geht gegen 100 %, die Test-Genauigkeit bleibt zurück.");
        combo(ui, "norm", "Normierung:", &mut c.norm, &Norm::ALL, |n| n.name());
        ui.add(egui::Slider::new(&mut c.shift, 0..=4).text("Augmentierung ±px"))
            .on_hover_text("Verschiebt jedes Trainingsbild zufällig um bis zu so viele Pixel (Datenaugmentierung).");
        ui.label(RichText::new(format!("Cache: {}", data::cache_dir().display())).small().color(MUTED));
    }

    fn cfg_arch(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Vorlage:");
            egui::ComboBox::from_id_salt("preset").selected_text("auswählen …").width(250.0).show_ui(ui, |ui| {
                for p in presets() {
                    if ui.selectable_label(false, p.name).on_hover_text(p.info).clicked() {
                        self.cfg.layers = p.layers.clone();
                        if let Some((o, lr, init)) = p.opt {
                            self.cfg.opt.kind = o;
                            self.cfg.lr = lr;
                            self.cfg.init = init;
                        }
                        self.preset_info = Some(p.info);
                    }
                }
            });
        });
        if let Some(i) = self.preset_info {
            ui.label(RichText::new(i).small().color(MUTED));
        }
        ui.add_space(4.0);
        let (shapes, params): (Vec<String>, Vec<usize>) = match &self.preview {
            Ok(m) => m.layers.iter().map(|l| (l.out_shape().describe(), l.n_params())).unzip(),
            Err(_) => (Vec::new(), Vec::new()),
        };
        ui.label(RichText::new(format!("Eingabe  1×{SIDE}×{SIDE}")).monospace().color(MUTED));
        let mut action: Option<(usize, i32)> = None;
        let n = self.cfg.layers.len();
        for i in 0..n {
            egui::Frame::group(ui.style()).inner_margin(4.0).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{:>2}", i + 1)).monospace().color(MUTED));
                    let spec = &mut self.cfg.layers[i];
                    match spec {
                        LayerSpec::Dense { units } => {
                            ui.label(RichText::new("Dense").strong().color(ACCENT));
                            ui.add(egui::DragValue::new(units).range(1..=4096).speed(2));
                        }
                        LayerSpec::Conv { filters, k, same } => {
                            ui.label(RichText::new("Conv").strong().color(Color32::from_rgb(230, 90, 160)));
                            ui.add(egui::DragValue::new(filters).range(1..=128).speed(0.3).suffix(" F"));
                            ui.add(egui::DragValue::new(k).range(1..=9).speed(0.1).prefix("k="));
                            ui.checkbox(same, "same");
                        }
                        LayerSpec::MaxPool { size } => {
                            ui.label(RichText::new("MaxPool").strong().color(Color32::from_rgb(60, 210, 200)));
                            ui.add(egui::DragValue::new(size).range(2..=4).speed(0.05));
                        }
                        LayerSpec::Act(a) => {
                            egui::ComboBox::from_id_salt(("act", i)).selected_text(RichText::new(a.name()).color(GREEN)).width(95.0).show_ui(ui, |ui| {
                                for k in ActKind::ALL {
                                    ui.selectable_value(a, k, k.name());
                                }
                            });
                        }
                        LayerSpec::Dropout { p } => {
                            ui.label(RichText::new("Dropout").strong());
                            ui.add(egui::DragValue::new(p).range(0.0..=0.9).speed(0.01).fixed_decimals(2));
                        }
                        LayerSpec::BatchNorm => {
                            ui.label(RichText::new("BatchNorm").strong().color(Color32::from_rgb(160, 120, 255)));
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("✕").on_hover_text("Entfernen").clicked() {
                            action = Some((i, 0));
                        }
                        if ui.add_enabled(i + 1 < n, egui::Button::new("⏷").small()).clicked() {
                            action = Some((i, 1));
                        }
                        if ui.add_enabled(i > 0, egui::Button::new("⏶").small()).clicked() {
                            action = Some((i, -1));
                        }
                    });
                });
                if let Some(s) = shapes.get(i) {
                    let p = params[i];
                    ui.label(RichText::new(if p > 0 { format!("     → {s}   ({} Parameter)", fmt_n(p)) } else { format!("     → {s}") }).monospace().small().color(MUTED));
                }
            });
        }
        match action {
            Some((i, 0)) => {
                self.cfg.layers.remove(i);
            }
            Some((i, d)) => self.cfg.layers.swap(i, (i as i32 + d) as usize),
            None => {}
        }
        if let Some(p) = params.last() {
            ui.label(RichText::new(format!("Ausgabe  Dense 10 → Softmax ({}, fest)", fmt_n(*p))).monospace().color(MUTED));
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Hinzufügen:");
            let l = &mut self.cfg.layers;
            if ui.small_button("+ Dense").clicked() {
                l.push(LayerSpec::Dense { units: 64 });
            }
            if ui.small_button("+ Aktivierung").clicked() {
                l.push(LayerSpec::Act(ActKind::ReLU));
            }
            if ui.small_button("+ Conv").clicked() {
                l.push(LayerSpec::Conv { filters: 8, k: 3, same: true });
            }
            if ui.small_button("+ MaxPool").clicked() {
                l.push(LayerSpec::MaxPool { size: 2 });
            }
            if ui.small_button("+ BatchNorm").clicked() {
                l.push(LayerSpec::BatchNorm);
            }
            if ui.small_button("+ Dropout").clicked() {
                l.push(LayerSpec::Dropout { p: 0.2 });
            }
        });
        match &self.preview {
            Ok(m) => {
                ui.label(RichText::new(format!("{} Parameter · {} MACs pro Bild", fmt_n(m.n_params()), fmt_n(m.macs()))).color(GREEN));
            }
            Err(e) if !e.is_empty() => {
                ui.label(RichText::new(e).color(RED));
            }
            _ => {}
        }
    }

    fn cfg_init(&mut self, ui: &mut egui::Ui) {
        let c = &mut self.cfg;
        combo(ui, "init", "Init:", &mut c.init, &Init::ALL, |i| i.name());
        combo(ui, "loss", "Verlust:", &mut c.loss, &[LossKind::CrossEntropy, LossKind::Mse], |l| l.name());
        ui.add(egui::Slider::new(&mut c.smoothing, 0.0..=0.3).text("Label Smoothing ε")).on_hover_text("Ziel = (1−ε)·One-Hot + ε/10");
    }

    fn cfg_opt(&mut self, ui: &mut egui::Ui) {
        let c = &mut self.cfg;
        if combo(ui, "opt", "Verfahren:", &mut c.opt.kind, &OptKind::ALL, |o| o.name()) {
            c.lr = c.opt.kind.default_lr();
        }
        ui.add(egui::Slider::new(&mut c.lr, 1e-5..=1.0).logarithmic(true).text("Lernrate η").custom_formatter(|v, _| format!("{v:.2e}")));
        match c.opt.kind {
            OptKind::Momentum | OptKind::Nesterov => {
                ui.add(egui::Slider::new(&mut c.opt.momentum, 0.0..=0.99).text("Momentum μ"));
            }
            OptKind::RmsProp => {
                ui.add(egui::Slider::new(&mut c.opt.beta2, 0.8..=0.9999).text("ρ (Abklingrate)"));
                ui.add(egui::Slider::new(&mut c.opt.eps, 1e-10..=1e-3).logarithmic(true).text("ε").custom_formatter(|v, _| format!("{v:.0e}")));
            }
            OptKind::Adam | OptKind::AdamW => {
                ui.add(egui::Slider::new(&mut c.opt.beta1, 0.0..=0.99).text("β₁"));
                ui.add(egui::Slider::new(&mut c.opt.beta2, 0.9..=0.9999).text("β₂"));
                ui.add(egui::Slider::new(&mut c.opt.eps, 1e-10..=1e-3).logarithmic(true).text("ε").custom_formatter(|v, _| format!("{v:.0e}")));
            }
            OptKind::Sgd => {}
        }
        let wd_label = if c.opt.kind == OptKind::AdamW { "Weight Decay (entkoppelt)" } else { "L2-Regularisierung λ" };
        ui.add(egui::Slider::new(&mut c.opt.weight_decay, 0.0..=0.1).logarithmic(true).smallest_positive(1e-6).text(wd_label).custom_formatter(|v, _| if v == 0.0 { "0".into() } else { format!("{v:.0e}") }));
        let mut clip = c.clip.is_some();
        ui.horizontal(|ui| {
            ui.checkbox(&mut clip, "Gradient Clipping (globale Norm)");
            let mut v = c.clip.unwrap_or(1.0);
            ui.add_enabled(clip, egui::DragValue::new(&mut v).range(0.01..=100.0).speed(0.05));
            c.clip = clip.then_some(v);
        });
    }

    fn cfg_sched(&mut self, ui: &mut egui::Ui) {
        let c = &mut self.cfg;
        combo(ui, "sched", "Plan:", &mut c.schedule, &Schedule::ALL, |s| s.name());
        match c.schedule {
            Schedule::Constant => {}
            Schedule::Step => {
                ui.horizontal(|ui| {
                    ui.label("alle");
                    ui.add(egui::DragValue::new(&mut c.step_every).range(1..=50));
                    ui.label("Epochen ×");
                    ui.add(egui::DragValue::new(&mut c.step_factor).range(0.01..=1.0).speed(0.01));
                });
            }
            Schedule::Exponential => {
                ui.add(egui::Slider::new(&mut c.exp_gamma, 0.5..=1.0).text("γ pro Epoche"));
            }
            Schedule::Cosine => {
                ui.add(egui::Slider::new(&mut c.lr_min_frac, 0.0..=0.5).text("Minimum (Anteil)"));
            }
        }
        ui.add(egui::Slider::new(&mut c.warmup_epochs, 0.0..=5.0).text("Warmup (Epochen)"));
        let pts: Vec<[f64; 2]> = (0..=200)
            .map(|i| {
                let e = i as f32 / 200.0 * c.epochs as f32;
                [e as f64, c.lr_at(e) as f64]
            })
            .collect();
        let mx = pts.iter().map(|p| p[1]).fold(0.0, f64::max);
        Plot::new("lrplan").height(90.0).allow_drag(false).allow_zoom(false).allow_scroll(false).include_y(0.0).include_y(mx * 1.15).x_axis_label("Epoche").show(ui, |p| {
            p.line(Line::new("η", PlotPoints::from(pts)).color(ACCENT));
        });
    }

    fn cfg_train(&mut self, ui: &mut egui::Ui) {
        let cores = self.cores;
        let c = &mut self.cfg;
        ui.add(egui::Slider::new(&mut c.epochs, 1..=100).text("Epochen"));
        ui.add(egui::Slider::new(&mut c.batch, 1..=1024).logarithmic(true).text("Batchgröße"));
        c.threads = c.threads.clamp(1, cores);
        ui.add(egui::Slider::new(&mut c.threads, 1..=cores).text(format!("CPU-Threads (von {cores})")))
            .on_hover_text("So viele Kerne rechnen parallel. In der Aktivitätsanzeige siehst du die Last, oben Durchsatz und GFLOP/s.");
        ui.horizontal(|ui| {
            ui.label("Seed:");
            ui.add(egui::DragValue::new(&mut c.seed));
            if ui.small_button("🎲").clicked() {
                c.seed = (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(1) % 1_000_000) as u64;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Test-Stichprobe alle");
            ui.add(egui::DragValue::new(&mut c.eval_every).range(0..=2000).speed(5));
            ui.label("Schritte");
        })
        .response
        .on_hover_text("Zwischenauswertung auf 1000 Testbildern; am Epochenende immer auf allen 10 000. 0 = aus.");
    }
}

// ---------------------------------------------------------------------------
// Diagramm-Hilfen
// ---------------------------------------------------------------------------

struct Series {
    name: String,
    pts: Vec<[f64; 2]>,
    color: Color32,
    width: f32,
    dots: bool,
}

fn line(name: impl Into<String>, pts: Vec<[f64; 2]>, color: Color32) -> Series {
    Series { name: name.into(), pts, color, width: 1.6, dots: false }
}

#[allow(clippy::too_many_arguments)]
fn chart(ui: &mut egui::Ui, id: &str, title: &str, height: f32, log: bool, series: Vec<Series>, hline: Option<(f64, &str)>, x_label: &str) {
    ui.label(RichText::new(title).strong());
    let mut p = Plot::new(id).height(height.max(80.0)).legend(Legend::default()).x_axis_label(x_label).allow_scroll(false);
    if log {
        p = p.y_axis_formatter(|m, _| viz::log_label(m.value));
    }
    p.show(ui, |pu| {
        for se in series {
            if se.pts.is_empty() {
                continue;
            }
            if se.dots {
                pu.points(Points::new(se.name.clone(), PlotPoints::from(se.pts.clone())).color(se.color).radius(3.0));
            }
            pu.line(Line::new(se.name, PlotPoints::from(se.pts)).color(se.color).width(se.width));
        }
        if let Some((y, name)) = hline {
            pu.hline(HLine::new(name, y).color(MUTED));
        }
    });
}

fn bar_probs(ui: &mut egui::Ui, probs: &[f32], names: &[&str; 10], truth: Option<usize>, width: f32) {
    let pred = (0..probs.len()).max_by(|&a, &b| probs[a].total_cmp(&probs[b])).unwrap_or(0);
    for (i, &p) in probs.iter().enumerate() {
        ui.horizontal(|ui| {
            let mut name = RichText::new(format!("{:>8}", names[i])).monospace();
            if Some(i) == truth {
                name = name.color(GREEN).strong();
            }
            ui.label(name);
            let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 14.0), Sense::hover());
            let painter = ui.painter();
            painter.rect_filled(rect, 2.0, Color32::from_gray(35));
            let mut r = rect;
            r.set_width(rect.width() * p.clamp(0.0, 1.0));
            let col = if i == pred { if truth.is_none_or(|t| t == i) { GREEN } else { RED } } else { BLUE };
            painter.rect_filled(r, 2.0, col);
            ui.label(RichText::new(format!("{:5.1} %", p * 100.0)).monospace().small());
        });
    }
}

fn histogram(vals: &[f32], bins: usize) -> Vec<Bar> {
    let mn = vals.iter().cloned().fold(f32::INFINITY, f32::min) as f64;
    let mx = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    if !mn.is_finite() || !mx.is_finite() {
        return Vec::new();
    }
    let w = ((mx - mn) / bins as f64).max(1e-12);
    let mut c = vec![0u32; bins];
    for &v in vals {
        c[(((v as f64 - mn) / w) as usize).min(bins - 1)] += 1;
    }
    c.iter().enumerate().map(|(i, &n)| Bar::new(mn + (i as f64 + 0.5) * w, n as f64).width(w * 0.95)).collect()
}

fn stats(v: &[f32]) -> String {
    let n = v.len().max(1) as f64;
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let sd = (v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
    let norm = v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    format!("n={}  μ={mean:+.2e}  σ={sd:.2e}  min={mn:+.2e}  max={mx:+.2e}  ‖·‖₂={norm:.3e}", v.len())
}

// ---------------------------------------------------------------------------
// Tab: Verlauf
// ---------------------------------------------------------------------------

impl App {
    fn tab_verlauf(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.smooth, 0.0..=0.995).text("Glättung (EMA)"));
            ui.checkbox(&mut self.log_loss, "Verlust logarithmisch");
        });
        let (a, lg) = (self.smooth as f64, self.log_loss);
        let pct = |v: Vec<[f64; 2]>| v.into_iter().map(|p| [p[0], p[1] * 100.0]).collect::<Vec<_>>();
        let (loss_raw, loss_s, val_loss, acc_s, val_acc, lr, gn, ep_tr, ep_te) = {
            let s = self.tr.shared.lock().unwrap();
            let h = &s.hist;
            (
                viz::curve(&h.loss, 0.0, 1500, lg),
                viz::curve(&h.loss, a, 1500, lg),
                viz::curve(&h.val_loss, 0.0, 1500, lg),
                pct(viz::curve(&h.acc, a, 1500, false)),
                pct(viz::curve(&h.val_acc, 0.0, 1500, false)),
                viz::curve(&h.lr, 0.0, 800, false),
                viz::curve(&h.grad_global, a, 1500, true),
                pct(h.epoch_train_acc.clone()),
                pct(h.epoch_val_acc.clone()),
            )
        };
        if loss_raw.is_empty() {
            ui.add_space(40.0);
            ui.vertical_centered(|ui| {
                ui.label(RichText::new("Noch kein Training gelaufen.").size(18.0));
                ui.label("Links Architektur und Parameter wählen, dann oben „Training starten“.");
                ui.label(RichText::new("Tipp: Der Tab „Netz & Aktivierungen“ zeigt das untrainierte Netz schon jetzt.").color(MUTED));
            });
            return;
        }
        let h = (ui.available_height() - 70.0) / 2.0 - 24.0;
        ui.columns(2, |c| {
            chart(
                &mut c[0],
                "loss",
                if lg { "Verlust (log)" } else { "Verlust" },
                h,
                lg,
                vec![
                    Series { name: "Train (roh)".into(), pts: loss_raw, color: Color32::from_rgba_unmultiplied(255, 150, 50, 45), width: 1.0, dots: false },
                    line("Train (geglättet)", loss_s, ACCENT),
                    Series { name: "Test".into(), pts: val_loss, color: BLUE, width: 2.0, dots: true },
                ],
                None,
                "Schritt",
            );
            chart(
                &mut c[1],
                "acc",
                "Genauigkeit [%]",
                h,
                false,
                vec![line("Train-Batches (geglättet)", acc_s, ACCENT), Series { name: "Test".into(), pts: val_acc, color: BLUE, width: 2.0, dots: true }],
                None,
                "Schritt",
            );
        });
        ui.columns(3, |c| {
            let lr_max = lr.iter().map(|p| p[1]).fold(0.0, f64::max);
            c[0].label(RichText::new("Lernrate η").strong());
            Plot::new("lr").height(h.max(80.0)).include_y(0.0).include_y(lr_max * 1.15).x_axis_label("Schritt").allow_scroll(false).show(&mut c[0], |p| {
                p.line(Line::new("η", PlotPoints::from(lr)).color(GREEN).width(1.6));
            });
            chart(&mut c[1], "gn", "Globale Gradientennorm ‖∇θ‖ (log)", h, true, vec![line("‖∇θ‖", gn, Color32::from_rgb(230, 90, 160))], None, "Schritt");
            let se = vec![
                Series { name: "Train-Acc".into(), pts: ep_tr, color: ACCENT, width: 2.0, dots: true },
                Series { name: "Test-Acc".into(), pts: ep_te, color: BLUE, width: 2.0, dots: true },
            ];
            chart(&mut c[2], "ep", "Pro Epoche: Train vs. Test [%]", h, false, se, None, "Epoche");
        });
    }
}

// ---------------------------------------------------------------------------
// Tab: Netz & Aktivierungen
// ---------------------------------------------------------------------------

impl App {
    fn probe_selector(&mut self, ui: &mut egui::Ui) {
        let ds = self.dataset();
        let wrong: Vec<usize> = self.tr.shared.lock().unwrap().last_eval.as_ref().map(|e| e.wrong.iter().map(|w| w.0).collect()).unwrap_or_default();
        ui.horizontal_wrapped(|ui| {
            let mut idx = match self.probe {
                Probe::Test(i) => i,
                Probe::Drawing => 0,
            };
            let n = ds.as_ref().map(|d| d.n_test()).unwrap_or(10_000);
            ui.label("Testbild");
            if ui.small_button("◀").clicked() {
                self.probe = Probe::Test((idx + n - 1) % n);
            }
            if ui.add(egui::DragValue::new(&mut idx).range(0..=n - 1)).changed() {
                self.probe = Probe::Test(idx);
            }
            if ui.small_button("▶").clicked() {
                self.probe = Probe::Test((idx + 1) % n);
            }
            if ui.small_button("🎲 Zufall").clicked() {
                let r = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_nanos() as usize).unwrap_or(0);
                self.probe = Probe::Test(r % n);
            }
            if !wrong.is_empty() && ui.small_button("nächster Fehler").on_hover_text("Springt zum nächsten falsch klassifizierten Testbild (nach Konfidenz sortiert).").clicked() {
                let pos = wrong.iter().position(|&w| Probe::Test(w) == self.probe).map(|p| (p + 1) % wrong.len()).unwrap_or(0);
                self.probe = Probe::Test(wrong[pos]);
            }
            if ui.selectable_label(self.probe == Probe::Drawing, "✏ eigene Zeichnung").clicked() {
                self.probe = Probe::Drawing;
            }
        });
    }

    fn tab_netz(&mut self, ui: &mut egui::Ui) {
        self.probe_selector(ui);
        let ds = self.dataset();
        let ctx = ui.ctx().clone();
        let probe = self.probe;
        let Some((tr, model, cfg, trained)) = self.trace() else {
            ui.label(self.load_msg.lock().unwrap().clone());
            ui.label("Datensatz wird geladen …");
            return;
        };
        let (key, img, probs, outs, input) = (tr.key, tr.img.clone(), tr.probs.clone(), tr.outs.clone(), tr.input.clone());
        let names = cfg.dataset.class_names();
        let truth = match (probe, &ds) {
            (Probe::Test(i), Some(d)) => Some(d.test_y[i.min(d.n_test() - 1)] as usize),
            _ => None,
        };
        if !trained {
            ui.label(RichText::new("Untrainiertes Netz (zufällige Startgewichte). Starte das Training und sieh zu, wie sich die Aktivierungen verändern.").color(ACCENT));
        }
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(300.0);
                ui.label(RichText::new("Eingabe").strong());
                let t = self.tex.get(&ctx, "probe_img", key, || viz::digit(&img, SIDE));
                image(ui, &t, 196.0, 196.0);
                let ti = self.tex.get(&ctx, "probe_in", key, || viz::vector(&input, SIDE, viz::Scale::auto(&input)));
                ui.horizontal(|ui| {
                    image(ui, &ti, 84.0, 84.0);
                    let mn = input.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mx = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    ui.label(RichText::new(format!("ins Netz (normiert)\n{}\nmin {mn:+.2}\nmax {mx:+.2}", cfg.norm.name())).small().color(MUTED));
                });
                if let Some(t) = truth {
                    ui.label(format!("Wahres Label: {}", names[t]));
                }
                let pred = (0..10).max_by(|&a, &b| probs[a].total_cmp(&probs[b])).unwrap_or(0);
                ui.label(RichText::new(format!("Vorhersage: {}  ({:.1} %)", names[pred], probs[pred] * 100.0)).size(17.0).strong().color(if truth.is_none_or(|t| t == pred) { GREEN } else { RED }));
                ui.add_space(6.0);
                ui.label(RichText::new("Softmax-Ausgabe").strong());
                bar_probs(ui, &probs, &names, truth, 120.0);
            });
            ui.separator();
            egui::ScrollArea::vertical().id_salt("layers").auto_shrink(false).show(ui, |ui| ui.vertical(|ui| {
                let avail = (ui.available_width() - 270.0).max(200.0);
                let mut shape = model.input;
                let nl = model.layers.len();
                for (li, (l, out)) in model.layers.iter().zip(&outs).enumerate() {
                    let os = l.out_shape();
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.horizontal_top(|ui| {
                            ui.vertical(|ui| {
                                ui.set_width(240.0);
                                let title = if li + 1 == nl { "Ausgabe (Logits)".to_string() } else { l.name() };
                                ui.label(RichText::new(format!("{}  {title}", li + 1)).strong().color(layer_color(l)));
                                ui.label(RichText::new(format!("{} → {}", shape.describe(), os.describe())).monospace().small());
                                if l.n_params() > 0 {
                                    ui.label(RichText::new(format!("{} Parameter, {} MACs", fmt_n(l.n_params()), fmt_n(l.macs()))).small().color(MUTED));
                                }
                                let n = out.len().max(1) as f32;
                                let mean = out.iter().sum::<f32>() / n;
                                let sd = (out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
                                let mut info = format!("μ={mean:+.3}  σ={sd:.3}");
                                if let Layer::Act(a) = l {
                                    if matches!(a.kind, ActKind::ReLU) {
                                        let zero = out.iter().filter(|&&v| v <= 0.0).count();
                                        info += &format!("\n{:.0} % der Einheiten = 0", zero as f32 / n * 100.0);
                                    }
                                }
                                ui.label(RichText::new(info).small().monospace().color(MUTED));
                            });
                            let img = act_image(out, os);
                            let (iw, ih) = (img.size[0] as f32, img.size[1] as f32);
                            let sc = (avail / iw).min(if os.spatial() { 4.0 } else { 14.0 }).min(340.0 / ih).max(0.5);
                            let t = self.tex.get(&ctx, &format!("act{li}"), key, || img);
                            image(ui, &t, iw * sc, ih * sc);
                        });
                    });
                    shape = os;
                }
                ui.label(RichText::new("Farben: orange = positiv, blau = negativ (falls negative Werte vorkommen), sonst Viridis von 0 bis Maximum. Jede Schicht ist für sich normiert.").small().color(MUTED));
            }));
        });
    }
}

fn layer_color(l: &Layer) -> Color32 {
    match l {
        Layer::Dense(_) => ACCENT,
        Layer::Conv(_) => Color32::from_rgb(230, 90, 160),
        Layer::Pool(_) => Color32::from_rgb(60, 210, 200),
        Layer::Act(_) => GREEN,
        Layer::Dropout(_) => MUTED,
        Layer::BatchNorm(_) => Color32::from_rgb(160, 120, 255),
    }
}

/// Aktivierungsbild: räumlich = eine Kachel pro Kanal, sonst ein Pixelraster.
fn act_image(out: &[f32], s: nn::Shape) -> ColorImage {
    let sc = viz::Scale::auto(out);
    if s.spatial() {
        viz::tiles(s.c, s.w, s.h, s.c.min(16), |_| sc, |t, x, y| out[t * s.h * s.w + y * s.w + x])
    } else {
        let cols = if s.c <= 16 { s.c } else { 32 };
        viz::vector(out, cols, sc)
    }
}

// ---------------------------------------------------------------------------
// Tab: Gewichte
// ---------------------------------------------------------------------------

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-8)
}

fn show_img(ui: &mut egui::Ui, tex: &mut Textures, ctx: &egui::Context, key: &str, ver: u64, img: ColorImage, max_w: f32, max_scale: f32) {
    let (w, h) = (img.size[0] as f32, img.size[1] as f32);
    let s = (max_w / w).min(max_scale).max(0.25);
    let t = tex.get(ctx, key, ver, || img);
    image(ui, &t, w * s, h * s);
}

impl App {
    fn tab_gewichte(&mut self, ui: &mut egui::Ui) {
        let Some((model, ver, _, trained)) = self.model() else {
            ui.label("Kein gültiges Netz.");
            return;
        };
        let ctx = ui.ctx().clone();
        let pl: Vec<usize> = model.layers.iter().enumerate().filter(|(_, l)| l.has_params()).map(|(i, _)| i).collect();
        if !pl.contains(&self.sel_layer) {
            self.sel_layer = pl[0];
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Schicht:");
            for &i in &pl {
                let name = if i + 1 == model.layers.len() { format!("{} Ausgabe", i + 1) } else { format!("{} {}", i + 1, model.layers[i].name()) };
                ui.selectable_value(&mut self.sel_layer, i, name);
            }
        });
        if !trained {
            ui.label(RichText::new("Untrainierte Startgewichte. Vergleiche die Initialisierungen He, Xavier, σ=0,01 und Null.").color(ACCENT));
        }
        let li = self.sel_layer;
        let layer = &model.layers[li];
        let ps = layer.params();
        let (w, g) = (&ps[0].w, &ps[0].g);
        let avail = ui.available_width() - 20.0;
        let top_h = (ui.available_height() * 0.58).max(200.0);
        let v = ver.wrapping_mul(131) ^ li as u64;
        egui::ScrollArea::vertical().id_salt("wts").max_height(top_h).auto_shrink([false, true]).show(ui, |ui| match layer {
            Layer::Dense(d) if d.inp == PIX => {
                ui.label(RichText::new(format!("Eingehende Gewichte jedes der {} Neuronen als 28×28-Bild: das Muster, auf das ein Neuron anspricht.", d.out)).color(MUTED));
                let n = d.out.min(256);
                let cols = if n <= 10 { n } else { ((avail / 64.0) as usize).clamp(4, 32) };
                let img = viz::tiles(n, SIDE, SIDE, cols, |t| viz::Scale::Signed(max_abs(&w[t * PIX..][..PIX])), |t, x, y| w[t * PIX + y * SIDE + x]);
                show_img(ui, &mut self.tex, &ctx, "w_dense", v, img, avail, 6.0);
                if d.out > 256 {
                    ui.label(RichText::new("(die ersten 256 Neuronen)").small().color(MUTED));
                }
            }
            Layer::Dense(d) => {
                ui.label(RichText::new(format!("Gewichtsmatrix W ({} × {}): Zeile = Ausgabeneuron, Spalte = Eingang.", d.out, d.inp)).color(MUTED));
                let sc = viz::Scale::Signed(max_abs(w));
                let img = ColorImage::new([d.inp, d.out], w.iter().map(|&x| sc.color(x)).collect());
                let s = (avail / d.inp as f32).clamp(0.2, 16.0);
                let h = (d.out as f32 * s).min(top_h - 40.0).max(d.out as f32 * 0.2);
                let t = self.tex.get(&ctx, "w_mat", v, || img);
                image(ui, &t, d.inp as f32 * s, h);
            }
            Layer::Conv(c) => {
                let cin = c.cin.min(16);
                ui.label(RichText::new(format!("{} Filter mit je {}×{}×{} Gewichten. Zeile = Ausgabekanal, Spalte = Eingangskanal{}.", c.cout, c.cin, c.k, c.k, if c.cin > 16 { " (erste 16)" } else { "" })).color(MUTED));
                let kk = c.k * c.k;
                let ckk = c.cin * kk;
                // Bei nur einem Eingangskanal: Filter nebeneinander statt untereinander
                let cols = if cin == 1 { c.cout.min(16) } else { cin };
                let img = viz::tiles(c.cout * cin, c.k, c.k, cols, |t| viz::Scale::Signed(max_abs(&w[(t / cin) * ckk..][..ckk])), |t, x, y| w[(t / cin) * ckk + (t % cin) * kk + y * c.k + x]);
                show_img(ui, &mut self.tex, &ctx, "w_conv", v, img, avail.min(cols as f32 * 110.0), 24.0);
            }
            Layer::BatchNorm(b) => {
                ui.label(RichText::new("BatchNorm: γ (Skalierung) und β (Verschiebung) pro Kanal bzw. Neuron.").color(MUTED));
                let gb: Vec<Bar> = b.gamma.w.iter().enumerate().map(|(i, &x)| Bar::new(i as f64, x as f64).width(0.8)).collect();
                let bb: Vec<Bar> = b.beta.w.iter().enumerate().map(|(i, &x)| Bar::new(i as f64, x as f64).width(0.8)).collect();
                Plot::new("bn").height(top_h - 60.0).legend(Legend::default()).show(ui, |p| {
                    p.bar_chart(BarChart::new("γ", gb).color(ACCENT));
                    p.bar_chart(BarChart::new("β", bb).color(BLUE));
                });
            }
            _ => {}
        });
        ui.separator();
        let hw = histogram(w, 60);
        let hg = histogram(g, 60);
        let h = (ui.available_height() - 50.0).max(110.0);
        ui.columns(2, |c| {
            c[0].label(RichText::new("Histogramm der Gewichte").strong());
            c[0].label(RichText::new(stats(w)).small().monospace().color(MUTED));
            Plot::new("hw").height(h).show(&mut c[0], |p| p.bar_chart(BarChart::new("W", hw).color(ACCENT)));
            c[1].label(RichText::new("Histogramm des letzten Gradienten ∂L/∂W").strong());
            c[1].label(RichText::new(stats(g)).small().monospace().color(MUTED));
            if g.iter().all(|&x| x == 0.0) {
                c[1].add_space(30.0);
                c[1].label(RichText::new("Noch kein Gradient: Er entsteht beim ersten Trainingsschritt (z. B. mit „Schrittweise“ und ⏭ 1).").color(MUTED));
            } else {
                Plot::new("hg").height(h).show(&mut c[1], |p| p.bar_chart(BarChart::new("∂L/∂W", hg).color(BLUE)));
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Tab: Gradienten-Dynamik
// ---------------------------------------------------------------------------

impl App {
    fn tab_dynamik(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.smooth, 0.0..=0.995).text("Glättung (EMA)"));
        });
        let a = self.smooth as f64;
        let (gn, ur, astd, dead) = {
            let s = self.tr.shared.lock().unwrap();
            let h = &s.hist;
            let names = &s.layer_names;
            let nl = names.len();
            let label = |i: usize| if i + 1 == nl { format!("{:02} Ausgabe", i + 1) } else { format!("{:02} {}", i + 1, names[i]) };
            let (mut gn, mut ur, mut astd, mut dead) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for &i in &s.param_layers {
                let c = viz::series_color(gn.len());
                gn.push(line(label(i), viz::curve(&h.grad_norm[i], a, 1200, true), c));
                ur.push(line(label(i), viz::curve(&h.upd_ratio[i], a, 1200, true), c));
            }
            let mut k = 0;
            for i in 0..nl {
                if !h.act_dead.get(i).is_some_and(|v| !v.is_empty()) {
                    continue;
                }
                let c = viz::series_color(k);
                k += 1;
                astd.push(line(label(i), viz::curve(&h.act_std[i], a.min(0.9), 1200, false), c));
                dead.push(line(label(i), viz::curve(&h.act_dead[i], a.min(0.9), 1200, false).into_iter().map(|p| [p[0], p[1] * 100.0]).collect(), c));
            }
            (gn, ur, astd, dead)
        };
        if gn.iter().all(|s| s.pts.is_empty()) {
            ui.add_space(40.0);
            ui.vertical_centered(|ui| {
                ui.label(RichText::new("Hier erscheinen nach dem Start die Gradienten pro Schicht.").size(18.0));
                ui.label("Gut zu beobachten mit der Vorlage „Tiefes Sigmoid-MLP“: Die Gradienten der vorderen Schichten sind um Größenordnungen kleiner.");
            });
            return;
        }
        let h = (ui.available_height() - 80.0) / 2.0 - 24.0;
        ui.columns(2, |c| {
            chart(&mut c[0], "gnl", "‖∂L/∂W‖ pro Schicht (log)", h, true, gn, None, "Schritt");
            chart(&mut c[1], "url", "Update-Verhältnis ‖ΔW‖ / ‖W‖ pro Schritt (log)", h, true, ur, Some((-3.0, "Faustregel ≈ 1e-3")), "Schritt");
        });
        ui.columns(2, |c| {
            chart(&mut c[0], "astd", "Standardabweichung der Aktivierungen (Trainingsbatch)", h, false, astd, None, "Schritt");
            chart(&mut c[1], "dead", "Inaktive (ReLU ≤ 0) bzw. gesättigte (Sigmoid, Tanh) Einheiten [%]", h, false, dead, None, "Schritt");
        });
    }
}

// ---------------------------------------------------------------------------
// Tab: Auswertung
// ---------------------------------------------------------------------------

impl App {
    fn tab_auswertung(&mut self, ui: &mut egui::Ui) {
        let (ev, kind) = {
            let s = self.tr.shared.lock().unwrap();
            (s.last_eval.clone(), s.cfg.dataset)
        };
        let Some(ev) = ev else {
            ui.add_space(40.0);
            ui.vertical_centered(|ui| ui.label(RichText::new("Die Auswertung auf allen 10 000 Testbildern erscheint nach der ersten Epoche.").size(18.0)));
            return;
        };
        let names = kind.class_names();
        let ds = self.dataset();
        let ctx = ui.ctx().clone();
        let errors: u32 = ev.confusion.iter().enumerate().map(|(i, r)| r.iter().sum::<u32>() - r[i]).sum();
        ui.label(RichText::new(format!("Testset: Genauigkeit {:.2} %  ·  Verlust {:.4}  ·  {errors} Fehler", ev.acc * 100.0, ev.loss)).size(16.0).strong());
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new("Konfusionsmatrix (Zeile = wahr, Spalte = vorhergesagt)").strong());
                let cell = 42.0;
                let lab = 66.0;
                let (rect, _) = ui.allocate_exact_size(egui::vec2(lab + cell * 10.0 + 70.0, 24.0 + cell * 10.0 + 4.0), Sense::hover());
                let p = ui.painter_at(rect);
                let o = rect.min + egui::vec2(lab, 24.0);
                let maxoff = ev.confusion.iter().enumerate().flat_map(|(i, r)| r.iter().enumerate().filter(move |(j, _)| *j != i).map(|(_, &v)| v)).max().unwrap_or(1).max(1);
                for i in 0..10 {
                    p.text(o + egui::vec2(-6.0, (i as f32 + 0.5) * cell), egui::Align2::RIGHT_CENTER, names[i], egui::FontId::proportional(12.0), MUTED);
                    p.text(o + egui::vec2((i as f32 + 0.5) * cell, -4.0), egui::Align2::CENTER_BOTTOM, short(names[i]), egui::FontId::proportional(11.0), MUTED);
                    let row: u32 = ev.confusion[i].iter().sum();
                    for j in 0..10 {
                        let v = ev.confusion[i][j];
                        let r = egui::Rect::from_min_size(o + egui::vec2(j as f32 * cell, i as f32 * cell), egui::vec2(cell - 2.0, cell - 2.0));
                        let col = if i == j {
                            let t = v as f32 / row.max(1) as f32;
                            Color32::from_rgb((20.0 + 40.0 * t) as u8, (40.0 + 130.0 * t) as u8, (30.0 + 40.0 * t) as u8)
                        } else if v == 0 {
                            Color32::from_gray(28)
                        } else {
                            let t = ((v as f32).ln_1p() / (maxoff as f32).ln_1p()).clamp(0.15, 1.0);
                            Color32::from_rgb((60.0 + 180.0 * t) as u8, (30.0 + 40.0 * t) as u8, (30.0 + 30.0 * t) as u8)
                        };
                        p.rect_filled(r, 3.0, col);
                        if v > 0 {
                            p.text(r.center(), egui::Align2::CENTER_CENTER, v.to_string(), egui::FontId::monospace(11.0), Color32::WHITE);
                        }
                    }
                    let rec = ev.confusion[i][i] as f32 / row.max(1) as f32;
                    p.text(o + egui::vec2(10.0 * cell + 6.0, (i as f32 + 0.5) * cell), egui::Align2::LEFT_CENTER, format!("{:.1} %", rec * 100.0), egui::FontId::monospace(11.0), if rec > 0.98 { GREEN } else { ACCENT });
                }
                ui.label(RichText::new("Rechts daneben: Recall (Trefferquote) pro Klasse.").small().color(MUTED));
            });
            ui.separator();
            ui.vertical(|ui| {
                ui.label(RichText::new(format!("Fehlklassifikationen: die {} überzeugtesten Irrtümer. Klick = im Netz ansehen.", ev.wrong.len().min(120))).strong());
                let Some(ds) = ds else { return };
                egui::ScrollArea::vertical().id_salt("wrong").auto_shrink(false).show(ui, |ui| {
                    let cols = ((ui.available_width() / 96.0) as usize).max(1);
                    egui::Grid::new("wg").spacing(egui::vec2(6.0, 6.0)).show(ui, |ui| {
                        for (k, &(idx, pred, truth, conf)) in ev.wrong.iter().take(120).enumerate() {
                            ui.vertical(|ui| {
                                let t = self.tex.get(&ctx, &format!("wrong{k}"), idx as u64 * 31 + 7, || viz::digit(ds.test_image(idx), SIDE));
                                let r = ui.add(egui::Image::new(egui::load::SizedTexture::new(t.id(), egui::vec2(84.0, 84.0))).sense(Sense::click()));
                                if r.on_hover_text(format!("Testbild {idx}")).clicked() {
                                    self.probe = Probe::Test(idx);
                                    self.tab = Tab::Netz;
                                }
                                ui.label(RichText::new(format!("{}→{} {:.0}%", short(names[truth as usize]), short(names[pred as usize]), conf * 100.0)).small().monospace());
                            });
                            if (k + 1) % cols == 0 {
                                ui.end_row();
                            }
                        }
                    });
                });
            });
        });
    }
}

fn short(s: &str) -> String {
    s.chars().take(5).collect()
}

// ---------------------------------------------------------------------------
// Tab: Zeichnen
// ---------------------------------------------------------------------------

impl App {
    fn tab_zeichnen(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.label(RichText::new("Male eine Ziffer mit der Maus. Sie wird wie MNIST aufbereitet (in 20×20 eingepasst, Schwerpunkt zentriert) und live klassifiziert.").color(MUTED));
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                let size = 420.0;
                let (resp, painter) = ui.allocate_painter(egui::vec2(size, size), Sense::drag());
                let rect = resp.rect;
                if resp.dragged() || resp.drag_started() {
                    if let Some(pos) = resp.interact_pointer_pos() {
                        let p = (pos - rect.min) / size * draw::N as f32;
                        self.canvas.stroke_to(p.x, p.y);
                    }
                } else {
                    self.canvas.last = None;
                }
                let cv = self.canvas.version;
                let buf = &self.canvas.buf;
                let t = self.tex.get(&ctx, "canvas", cv, || ColorImage::new([draw::N, draw::N], buf.iter().map(|&v| viz::gray(v)).collect()));
                painter.image(t.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
                painter.rect_stroke(rect, 4.0, egui::Stroke::new(1.0, MUTED), egui::StrokeKind::Outside);
                ui.horizontal(|ui| {
                    if ui.button("🗑 Löschen").clicked() {
                        self.canvas.clear();
                    }
                    ui.add(egui::Slider::new(&mut self.canvas.brush, 5.0..=20.0).text("Pinsel"));
                });
                if ui.button("🔍 Durchs Netz verfolgen (Tab „Netz & Aktivierungen“)").clicked() {
                    self.probe = Probe::Drawing;
                    self.tab = Tab::Netz;
                }
            });
            ui.separator();
            ui.vertical(|ui| {
                if self.canvas.is_empty() {
                    ui.label("Noch nichts gezeichnet.");
                    return;
                }
                let prev = self.probe;
                self.probe = Probe::Drawing;
                let res = self.trace().map(|(t, _, cfg, trained)| (t.img.clone(), t.probs.clone(), cfg.dataset, trained, t.key));
                self.probe = prev;
                let Some((img, probs, kind, trained, key)) = res else { return };
                ui.label(RichText::new("Aufbereitet (28×28)").strong());
                let t = self.tex.get(&ctx, "draw28", key, || viz::digit(&img, SIDE));
                image(ui, &t, 168.0, 168.0);
                if !trained {
                    ui.label(RichText::new("Das Netz ist noch untrainiert.").color(ACCENT));
                }
                if kind == DatasetKind::Fashion {
                    ui.label(RichText::new("Hinweis: Das Netz wurde auf Fashion-MNIST trainiert.").color(ACCENT));
                }
                let names = kind.class_names();
                let pred = (0..10).max_by(|&a, &b| probs[a].total_cmp(&probs[b])).unwrap_or(0);
                ui.label(RichText::new(format!("Vorhersage: {}", names[pred])).size(26.0).strong().color(GREEN));
                bar_probs(ui, &probs, &names, None, 220.0);
            });
        });
    }
}
