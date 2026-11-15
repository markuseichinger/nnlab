//! nnlab: neuronale Netze konfigurieren, trainieren und beim Lernen zusehen.

mod advisor;
mod draw;
mod presets;
mod tutorial;
mod viz;

use advisor::{Advice, DataInfo, Level};
use eframe::egui::{self, Color32, ColorImage, RichText, Sense, TextureHandle, TextureOptions};
use egui_plot::{Bar, BarChart, HLine, Legend, Line, Plot, PlotPoints, Points};
use nnlab::data::{self, Augment, DataSpec, Dataset, Norm, Source, HF_SUGGESTIONS};
use nnlab::nn::{self, ActKind, Init, Layer, LayerSpec, LossKind, Model, OptKind, Shape};
use nnlab::trainer::{self, Schedule, Status, TrainConfig, Trainer};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tutorial::Topic;

fn main() -> eframe::Result {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("nnlab – Neuronale Netze trainieren").with_inner_size([1680.0, 1000.0]).with_min_inner_size([1150.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native("nnlab", opts, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

const ACCENT: Color32 = Color32::from_rgb(255, 150, 50);
const BLUE: Color32 = Color32::from_rgb(70, 160, 255);
const GREEN: Color32 = Color32::from_rgb(120, 209, 81);
const RED: Color32 = Color32::from_rgb(240, 90, 80);
const VIOLET: Color32 = Color32::from_rgb(160, 120, 255);
const MUTED: Color32 = Color32::from_rgb(140, 145, 155);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Verlauf,
    Netz,
    Gewichte,
    Dynamik,
    Auswertung,
    Zeichnen,
    Berater,
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

/// Setzt das Tutorial-Thema, wenn das Bedienelement angeklickt oder verändert wurde.
fn tp(slot: &mut Topic, r: egui::Response, t: Topic) -> egui::Response {
    if r.clicked() || r.drag_started() || r.gained_focus() || r.changed() {
        *slot = t;
    }
    r
}

/// Anklickbare Beschriftung, die ein Tutorial-Thema setzt.
fn tlabel(ui: &mut egui::Ui, slot: &mut Topic, text: impl Into<egui::WidgetText>, t: Topic) {
    let r = ui.add(egui::Label::new(text).sense(Sense::click()));
    tp(slot, r.on_hover_cursor(egui::CursorIcon::Help), t);
}

fn opt_topic(k: OptKind) -> Topic {
    match k {
        OptKind::Sgd => Topic::Sgd,
        OptKind::Momentum | OptKind::Nesterov => Topic::Momentum,
        OptKind::RmsProp => Topic::RmsProp,
        OptKind::Adam | OptKind::AdamW => Topic::Adam,
    }
}

fn layer_topic(l: &LayerSpec) -> Topic {
    match l {
        LayerSpec::Dense { .. } => Topic::Dense,
        LayerSpec::Conv { .. } => Topic::Conv,
        LayerSpec::MaxPool { .. } => Topic::MaxPool,
        LayerSpec::Act(_) => Topic::Activation,
        LayerSpec::Dropout { .. } => Topic::Dropout,
        LayerSpec::BatchNorm => Topic::BatchNorm,
    }
}

pub fn fmt_n(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2} M", n as f64 / 1e6)
    } else if n >= 10_000 {
        format!("{:.1} k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
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
    preview_key: Option<(Vec<LayerSpec>, Init, u64, (usize, usize, usize), usize)>,
    preview_ver: u64,
    trace: Option<Trace>,
    load_msg: Arc<Mutex<String>>,
    loading: Arc<AtomicBool>,
    lr_factor: f32,
    cores: usize,
    info_line: Option<String>,
    topic: Topic,
    show_tutorial: bool,
    hf_id: String,
    hf_side: usize,
    hf_rgb: bool,
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
            info_line: None,
            topic: Topic::Welcome,
            show_tutorial: true,
            hf_id: String::new(),
            hf_side: 32,
            hf_rgb: true,
        };
        app.ensure_data();
        app
    }

    fn dataset(&self) -> Option<Arc<Dataset>> {
        self.tr.data.lock().unwrap().clone()
    }

    /// Der geladene Datensatz, falls er zur aktuellen Auswahl passt.
    fn current_data(&self) -> Option<Arc<Dataset>> {
        self.dataset().filter(|d| d.spec == self.cfg.dataset)
    }

    fn set_dataset(&mut self, spec: DataSpec) {
        if spec != self.cfg.dataset {
            self.cfg.dataset = spec;
            self.load_msg.lock().unwrap().clear();
            self.probe = Probe::Test(0);
        }
    }

    /// Lädt den gewählten Datensatz im Hintergrund, falls nötig.
    fn ensure_data(&mut self) {
        if self.current_data().is_some() || self.loading.load(Ordering::SeqCst) || self.tr.running() {
            return;
        }
        if self.load_msg.lock().unwrap().starts_with("Fehler") {
            return;
        }
        self.loading.store(true, Ordering::SeqCst);
        let (data, msg, loading, ctx, spec) = (self.tr.data.clone(), self.load_msg.clone(), self.loading.clone(), self.ctx.clone(), self.cfg.dataset.clone());
        std::thread::spawn(move || {
            let r = trainer::ensure_data(&spec, &data, &|m| {
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

    fn n_classes(&self) -> usize {
        self.current_data().map(|d| d.n_classes()).unwrap_or(10)
    }

    fn update_preview(&mut self) {
        let key = (self.cfg.layers.clone(), self.cfg.init, self.cfg.seed, self.cfg.dataset.shape(), self.n_classes());
        if self.preview_key.as_ref() != Some(&key) {
            self.preview = Model::build(&self.cfg.layers, self.cfg.input_shape(), key.4, self.cfg.init, self.cfg.seed).map(Arc::new);
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
    fn trace(&mut self) -> Result<(&Trace, Arc<Model>, TrainConfig, bool, Arc<Dataset>), String> {
        let (model, ver, cfg, trained) = self.model().ok_or("Kein gültiges Netz (siehe Architektur).")?;
        let ds = self.dataset().filter(|d| d.spec == cfg.dataset).ok_or_else(|| {
            if trained {
                format!("Das Modell wurde auf „{}“ trainiert, dieser Datensatz ist gerade nicht geladen. Zurücksetzen oder neu trainieren.", cfg.dataset.label())
            } else {
                "Datensatz wird geladen …".to_string()
            }
        })?;
        if self.probe == Probe::Drawing && model.input != Shape::new(1, 28, 28) {
            return Err("Zeichnungen gibt es nur für Netze mit 1×28×28-Eingabe.".into());
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (ver, self.probe, cfg.norm as u8, &ds.spec).hash(&mut h);
        if self.probe == Probe::Drawing {
            self.canvas.version.hash(&mut h);
        }
        let key = h.finish();
        if self.trace.as_ref().map(|t| t.key) != Some(key) {
            let img = match self.probe {
                Probe::Test(i) => ds.test_image(i.min(ds.n_test() - 1)).to_vec(),
                Probe::Drawing => self.canvas.to_mnist(),
            };
            let mut input = vec![0.0; ds.img_len()];
            data::prepare(&img, &mut input, &ds, cfg.norm, Augment::default(), None);
            let mut m = (*model).clone();
            let outs = m.forward_trace(&input);
            let z = outs.last().ok_or("leeres Netz")?;
            let mut probs = vec![0.0; z.len()];
            nn::softmax(z, &mut probs);
            self.trace = Some(Trace { key, img, input, outs, probs });
        }
        Ok((self.trace.as_ref().unwrap(), model, cfg, trained, ds))
    }

    fn data_info(&self) -> DataInfo {
        let d = self.current_data();
        DataInfo { kind: self.cfg.dataset.kind(), shape: self.cfg.dataset.shape(), n_train: d.as_ref().map(|d| d.n_train()), classes: d.as_ref().map(|d| d.n_classes()) }
    }

    fn apply_recipe(&mut self, name: &str) {
        if let Some(r) = presets::recipes().into_iter().find(|r| r.name == name) {
            (r.apply)(&mut self.cfg);
            self.info_line = Some(format!("Voreinstellung „{}“ übernommen. {}", r.name, r.info));
            self.topic = Topic::Recipes;
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.update_preview();
        if !self.tr.running() {
            self.ensure_data();
        }
        egui::Panel::top("top").show(ui, |ui| self.top_bar(ui));
        egui::Panel::left("cfg").resizable(true).default_size(400.0).min_size(340.0).show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| self.config_ui(ui));
        });
        if self.show_tutorial {
            egui::Panel::right("tutorial").resizable(true).default_size(380.0).min_size(300.0).show(ui, |ui| {
                egui::ScrollArea::vertical().id_salt("tut").auto_shrink(false).show(ui, |ui| self.tutorial_ui(ui));
            });
        }
        egui::CentralPanel::default_margins().show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for (t, name, topic) in [
                    (Tab::Verlauf, "📈 Verlauf", Topic::ViewLoss),
                    (Tab::Netz, "🔍 Netz & Aktivierungen", Topic::ViewActivations),
                    (Tab::Gewichte, "▦ Gewichte", Topic::ViewWeights),
                    (Tab::Dynamik, "〰 Gradienten-Dynamik", Topic::ViewGradNorm),
                    (Tab::Auswertung, "🎯 Auswertung", Topic::ViewConfusion),
                    (Tab::Zeichnen, "✏ Zeichnen", Topic::ViewDrawing),
                    (Tab::Berater, "🧭 Berater", Topic::Advisor),
                ] {
                    let r = ui.selectable_value(&mut self.tab, t, RichText::new(name).size(15.0));
                    tp(&mut self.topic, r, topic);
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
                Tab::Berater => self.tab_berater(ui),
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
                let ok = self.preview.is_ok() && self.current_data().is_some();
                if ui.add_enabled(ok, big("▶ Training starten")).on_hover_text("Baut das Netz neu (mit dem Seed) und trainiert.").clicked() {
                    self.start(false);
                }
                let r = ui.add_enabled(ok, big("⏭ Schrittweise")).on_hover_text("Startet pausiert. Dann trainiert jeder Klick auf „⏭ 1“ genau einen Batch.");
                if tp(&mut self.topic, r, Topic::StepMode).clicked() {
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
                        let r = ui.add(big(&format!("⏭ {n}"))).on_hover_text(format!("{n} Batch(es) trainieren"));
                        if tp(&mut self.topic, r, Topic::StepMode).clicked() {
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
                tlabel(ui, &mut self.topic, "Lernraten-Faktor live:", Topic::LiveLr);
                let r = ui.add(egui::Slider::new(&mut self.lr_factor, 0.01..=10.0).logarithmic(true).fixed_decimals(2)).on_hover_text("Multipliziert die Lernrate sofort, ohne Neustart.");
                if tp(&mut self.topic, r, Topic::LiveLr).changed() {
                    self.tr.ctl.lr_scale_milli.store((self.lr_factor * 1000.0) as u32, Ordering::SeqCst);
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let label = if self.show_tutorial { "🎓 Tutorial an" } else { "🎓 Tutorial" };
                if ui.add(egui::Button::new(RichText::new(label).size(15.0)).selected(self.show_tutorial)).on_hover_text("Erklärungen zu jedem Parameter: einfach anklicken.").clicked() {
                    self.show_tutorial = !self.show_tutorial;
                }
                ui.separator();
                let mut d = self.tr.ctl.delay_ms.load(Ordering::Relaxed) as f32;
                let r = ui
                    .add(egui::Slider::new(&mut d, 0.0..=500.0).custom_formatter(|v, _| if v < 1.0 { "aus".into() } else { format!("{v:.0} ms") }))
                    .on_hover_text("Pause nach jedem Trainingsschritt, um dem Netz beim Lernen zuzusehen. Zählt nicht zur Rechenzeit.");
                if tp(&mut self.topic, r, Topic::SlowMo).changed() {
                    self.tr.ctl.delay_ms.store(d as u32, Ordering::Relaxed);
                }
                tlabel(ui, &mut self.topic, "🐢 Zeitlupe", Topic::SlowMo);
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
                ui.label(RichText::new(lm.lines().next().unwrap_or("")).color(if self.loading.load(Ordering::SeqCst) { BLUE } else { RED }));
            }
            if i.spe > 0 {
                let in_ep = (i.step as usize).saturating_sub(i.epoch * i.spe).min(i.spe);
                let frac = ((i.epoch as f32 + in_ep as f32 / i.spe as f32) / i.epochs.max(1) as f32).min(1.0);
                ui.add(egui::ProgressBar::new(frac).desired_width(150.0).text(format!("Epoche {}/{}", (i.epoch + 1).min(i.epochs), i.epochs)));
                let topic = &mut self.topic;
                let mut m = |ui: &mut egui::Ui, k: &str, v: String, t: Topic| {
                    tlabel(ui, topic, RichText::new(k).color(MUTED), t);
                    ui.label(RichText::new(v).strong().monospace());
                };
                m(ui, "Schritt", format!("{}", i.step), Topic::BatchSize);
                if let Some(l) = i.loss {
                    m(ui, "Verlust", format!("{l:.4}"), Topic::ViewLoss);
                }
                if let Some(a) = i.acc {
                    m(ui, "Train", format!("{:.2} %", a * 100.0), Topic::ViewAcc);
                }
                if let Some(a) = i.val_acc {
                    m(ui, "Test", format!("{:.2} %", a * 100.0), Topic::ViewAcc);
                }
                ui.separator();
                m(ui, "Durchsatz", format!("{:.0} Bilder/s", i.sps), Topic::Throughput);
                m(ui, "Leistung", format!("{:.1} GFLOP/s", i.gflops), Topic::Throughput);
                m(ui, "Threads", format!("{}", i.threads), Topic::Threads);
                m(ui, "Rechenzeit", format!("{:.1} s", i.elapsed.as_secs_f32()), Topic::Throughput);
                ui.separator();
                m(ui, "Parameter", fmt_n(i.n_params), Topic::Architecture);
                m(ui, "MACs/Bild", fmt_n(i.macs), Topic::Architecture);
            }
        });
        ui.add_space(2.0);
    }
}

// ---------------------------------------------------------------------------
// Konfiguration
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn combo<T: PartialEq + Copy>(ui: &mut egui::Ui, id: &str, label: &str, v: &mut T, all: &[T], name: impl Fn(T) -> &'static str, slot: &mut Topic, topic: Topic) -> bool {
    let before = *v;
    ui.horizontal(|ui| {
        tlabel(ui, slot, label, topic);
        let r = egui::ComboBox::from_id_salt(id).selected_text(name(*v)).width(220.0).show_ui(ui, |ui| {
            for &o in all {
                ui.selectable_value(v, o, name(o));
            }
        });
        tp(slot, r.response, topic);
    });
    before != *v
}

/// Abschnitt mit anklickbarer Überschrift (setzt das Tutorial-Thema).
fn section(ui: &mut egui::Ui, slot: &mut Topic, title: &str, topic: Topic, open: bool, body: impl FnOnce(&mut egui::Ui, &mut Topic)) {
    let mut t = *slot;
    let r = egui::CollapsingHeader::new(RichText::new(title).strong().size(14.5)).default_open(open).show(ui, |ui| body(ui, &mut t));
    *slot = t;
    if r.header_response.clicked() {
        *slot = topic;
    }
}

impl App {
    fn config_ui(&mut self, ui: &mut egui::Ui) {
        let running = self.tr.running();
        if running {
            ui.label(RichText::new("Während des Trainings gesperrt (bis auf Lernraten-Faktor und Zeitlupe oben).").color(MUTED).italics());
        }
        // Kurzfassung des Beraters
        let advice = advisor::check_config(&self.cfg, &self.data_info(), &self.preview, self.cores);
        let (errs, warns) = (advice.iter().filter(|a| a.level == Level::Error).count(), advice.iter().filter(|a| a.level == Level::Warn).count());
        let (txt, col) = match (errs, warns) {
            (0, 0) => ("🧭 Berater: keine Probleme".to_string(), GREEN),
            (0, w) => (format!("🧭 Berater: {w} Hinweis{} prüfen", if w == 1 { "" } else { "e" }), ACCENT),
            (e, w) => (format!("🧭 Berater: {e} Fehler, {w} Warnung{}", if w == 1 { "" } else { "en" }), RED),
        };
        if ui.add(egui::Button::new(RichText::new(txt).color(col).strong()).frame(true)).on_hover_text("Öffnet den Berater mit Details und Korrekturen.").clicked() {
            self.tab = Tab::Berater;
            self.topic = Topic::Advisor;
        }
        if let Some(l) = &self.info_line {
            ui.label(RichText::new(l).small().color(GREEN));
        }
        ui.add_enabled_ui(!running, |ui| {
            let mut topic = self.topic;
            section(ui, &mut topic, "Voreinstellungen", Topic::Recipes, true, |ui, t| self.cfg_recipes(ui, t));
            section(ui, &mut topic, "Datensatz", Topic::Dataset, true, |ui, t| self.cfg_data(ui, t));
            section(ui, &mut topic, "Architektur", Topic::Architecture, true, |ui, t| self.cfg_arch(ui, t));
            section(ui, &mut topic, "Initialisierung & Verlust", Topic::Init, true, |ui, t| self.cfg_init(ui, t));
            section(ui, &mut topic, "Optimierer", Topic::Optimizer, true, |ui, t| self.cfg_opt(ui, t));
            section(ui, &mut topic, "Lernraten-Plan", Topic::Schedule, true, |ui, t| self.cfg_sched(ui, t));
            section(ui, &mut topic, "Training & Rechenleistung", Topic::Epochs, true, |ui, t| self.cfg_train(ui, t));
            self.topic = topic;
        });
    }

    fn cfg_recipes(&mut self, ui: &mut egui::Ui, t: &mut Topic) {
        ui.label(RichText::new("Setzen die komplette Konfiguration (Datensatz und Threads bleiben).").small().color(MUTED));
        let mut chosen = None;
        ui.horizontal(|ui| {
            let r = egui::ComboBox::from_id_salt("recipe").selected_text("Empfohlen …").width(170.0).show_ui(ui, |ui| {
                for r in presets::recipes().iter().filter(|r| r.group == presets::Group::Recommended) {
                    if ui.selectable_label(false, r.name).on_hover_text(r.info).clicked() {
                        chosen = Some(r.name);
                    }
                }
            });
            tp(t, r.response, Topic::Recipes);
            let r = egui::ComboBox::from_id_salt("lesson").selected_text("Lehrbeispiel …").width(170.0).show_ui(ui, |ui| {
                for r in presets::recipes().iter().filter(|r| r.group == presets::Group::Lesson) {
                    if ui.selectable_label(false, r.name).on_hover_text(r.info).clicked() {
                        chosen = Some(r.name);
                    }
                }
            });
            tp(t, r.response, Topic::Recipes);
        });
        if let Some(n) = chosen {
            self.apply_recipe(n);
            *t = Topic::Recipes;
        }
    }

    fn cfg_data(&mut self, ui: &mut egui::Ui, t: &mut Topic) {
        let cur = self.cfg.dataset.clone();
        let mut pick: Option<DataSpec> = None;
        ui.horizontal_wrapped(|ui| {
            for (spec, name) in [(DataSpec::mnist(), "MNIST"), (DataSpec::fashion(), "Fashion-MNIST")] {
                let r = ui.selectable_label(cur == spec, name);
                if tp(t, r, Topic::Dataset).clicked() {
                    pick = Some(spec);
                }
            }
        });
        tlabel(ui, t, RichText::new("Von Hugging Face:").color(MUTED), Topic::HfImport);
        ui.horizontal_wrapped(|ui| {
            for s in HF_SUGGESTIONS.iter() {
                let spec = DataSpec::hf(s.id, s.side, s.rgb);
                let r = ui.selectable_label(cur == spec, s.name).on_hover_text(format!("{}\n{}", s.id, s.info));
                if tp(t, r, Topic::Dataset).clicked() {
                    pick = Some(spec);
                }
            }
        });
        egui::CollapsingHeader::new("Eigener Hugging-Face-Datensatz").default_open(matches!(&cur.source, Source::Hf(id) if !HF_SUGGESTIONS.iter().any(|s| s.id == id))).show(ui, |ui| {
            ui.horizontal(|ui| {
                tlabel(ui, t, "ID:", Topic::HfImport);
                let r = ui.add(egui::TextEdit::singleline(&mut self.hf_id).hint_text("besitzer/name").desired_width(200.0));
                tp(t, r, Topic::HfImport);
            });
            ui.horizontal(|ui| {
                tlabel(ui, t, "Größe:", Topic::HfImport);
                let r = ui.add(egui::DragValue::new(&mut self.hf_side).range(8..=128).suffix(" px"));
                tp(t, r, Topic::HfImport);
                let r = ui.checkbox(&mut self.hf_rgb, "Farbe (RGB)");
                tp(t, r, Topic::HfImport);
                let r = ui.add_enabled(self.hf_id.contains('/'), egui::Button::new("Laden"));
                if tp(t, r, Topic::HfImport).clicked() {
                    pick = Some(DataSpec::hf(&self.hf_id, self.hf_side, self.hf_rgb));
                }
            });
            ui.label(RichText::new("Jeder Datensatz mit einer Image- und einer ClassLabel-Spalte, z. B. von huggingface.co/datasets?task_categories=image-classification").small().color(MUTED));
        });
        if let Some(p) = pick {
            if let Source::Hf(id) = &p.source {
                self.hf_id = id.clone();
                self.hf_side = p.side;
                self.hf_rgb = p.rgb;
            }
            self.set_dataset(p);
        }
        // Status und Vorschau
        let ctx = ui.ctx().clone();
        match self.current_data() {
            Some(d) => {
                ui.label(RichText::new(format!("✓ {}: {} Train / {} Test · {} Klassen · {}×{}×{}", d.name, fmt_n(d.n_train()), fmt_n(d.n_test()), d.n_classes(), d.c, d.h, d.w)).color(GREEN).small());
                let names: Vec<&str> = d.classes.iter().map(|s| s.as_str()).take(26).collect();
                ui.label(RichText::new(format!("Klassen: {}{}", names.join(", "), if d.n_classes() > 26 { " …" } else { "" })).small().color(MUTED));
                let n = 16.min(d.n_train());
                let key = format!("strip_{:?}", d.spec);
                let mut h = std::collections::hash_map::DefaultHasher::new();
                d.spec.hash(&mut h);
                let img = self.tex.get(&ctx, "datastrip", h.finish(), || {
                    let (c, hh, w) = (d.c, d.h, d.w);
                    let cols = 8;
                    let rows = n.div_ceil(cols);
                    let mut px = vec![viz::GAP; (cols * (w + 1) + 1) * (rows * (hh + 1) + 1)];
                    let iw = cols * (w + 1) + 1;
                    for k in 0..n {
                        let s = viz::sample(d.train_image(k * (d.n_train() / n).max(1)), c, hh, w);
                        let (ox, oy) = (1 + (k % cols) * (w + 1), 1 + (k / cols) * (hh + 1));
                        for y in 0..hh {
                            for x in 0..w {
                                px[(oy + y) * iw + ox + x] = s.pixels[y * w + x];
                            }
                        }
                    }
                    ColorImage::new([iw, rows * (hh + 1) + 1], px)
                });
                let _ = key;
                let sz = img.size_vec2();
                let sc = (ui.available_width() / sz.x).min(2.0);
                image(ui, &img, sz.x * sc, sz.y * sc);
            }
            None => {
                let lm = self.load_msg.lock().unwrap().clone();
                if lm.starts_with("Fehler") {
                    ui.label(RichText::new(lm).color(RED).small());
                    if ui.small_button("Erneut versuchen").clicked() {
                        self.load_msg.lock().unwrap().clear();
                    }
                } else {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(RichText::new(if lm.is_empty() { "Lade Datensatz …".to_string() } else { lm }).color(BLUE).small());
                    });
                }
            }
        }
        ui.separator();
        let max_train = self.current_data().map(|d| d.n_train()).unwrap_or(60_000);
        let c = &mut self.cfg;
        c.train_size = c.train_size.min(max_train).max(100.min(max_train));
        ui.horizontal(|ui| {
            tlabel(ui, t, "Trainingsbeispiele:", Topic::TrainSize);
            let r = ui.add(egui::DragValue::new(&mut c.train_size).range(100..=max_train).speed(100));
            tp(t, r, Topic::TrainSize);
            if ui.small_button("alle").clicked() {
                c.train_size = max_train;
                *t = Topic::TrainSize;
            }
        });
        combo(ui, "norm", "Normierung:", &mut c.norm, &Norm::ALL, |n| n.name(), t, Topic::Normalization);
        ui.horizontal(|ui| {
            tlabel(ui, t, "Augmentierung:", Topic::Augmentation);
            let r = ui.add(egui::Slider::new(&mut c.aug.shift, 0..=4).suffix(" px"));
            tp(t, r, Topic::Augmentation);
            let r = ui.checkbox(&mut c.aug.flip, "spiegeln");
            tp(t, r, Topic::Augmentation);
        });
        ui.label(RichText::new(format!("Cache: {}", data::cache_dir().display())).small().color(MUTED));
    }

    fn cfg_arch(&mut self, ui: &mut egui::Ui, t: &mut Topic) {
        let mut chosen = None;
        ui.horizontal(|ui| {
            tlabel(ui, t, "Nur Architektur:", Topic::Architecture);
            let r = egui::ComboBox::from_id_salt("arch").selected_text("Vorlage …").width(200.0).show_ui(ui, |ui| {
                for p in presets::arch_presets() {
                    if ui.selectable_label(false, p.name).on_hover_text(p.info).clicked() {
                        chosen = Some(p.layers.clone());
                    }
                }
            });
            tp(t, r.response, Topic::Architecture);
        });
        if let Some(l) = chosen {
            self.cfg.layers = l;
        }
        ui.add_space(2.0);
        let (shapes, params): (Vec<String>, Vec<usize>) = match &self.preview {
            Ok(m) => m.layers.iter().map(|l| (l.out_shape().describe(), l.n_params())).unzip(),
            Err(_) => (Vec::new(), Vec::new()),
        };
        let (ic, ih, iw) = self.cfg.dataset.shape();
        ui.label(RichText::new(format!("Eingabe  {ic}×{ih}×{iw}")).monospace().color(MUTED));
        let mut action: Option<(usize, i32)> = None;
        let n = self.cfg.layers.len();
        for i in 0..n {
            egui::Frame::group(ui.style()).inner_margin(4.0).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{:>2}", i + 1)).monospace().color(MUTED));
                    let spec = &mut self.cfg.layers[i];
                    let topic = layer_topic(spec);
                    match spec {
                        LayerSpec::Dense { units } => {
                            tlabel(ui, t, RichText::new("Dense").strong().color(ACCENT), topic);
                            tp(t, ui.add(egui::DragValue::new(units).range(1..=4096).speed(2)), topic);
                        }
                        LayerSpec::Conv { filters, k, same } => {
                            tlabel(ui, t, RichText::new("Conv").strong().color(Color32::from_rgb(230, 90, 160)), topic);
                            tp(t, ui.add(egui::DragValue::new(filters).range(1..=256).speed(0.3).suffix(" F")), topic);
                            tp(t, ui.add(egui::DragValue::new(k).range(1..=9).speed(0.1).prefix("k=")), topic);
                            tp(t, ui.checkbox(same, "same"), topic);
                        }
                        LayerSpec::MaxPool { size } => {
                            tlabel(ui, t, RichText::new("MaxPool").strong().color(Color32::from_rgb(60, 210, 200)), topic);
                            tp(t, ui.add(egui::DragValue::new(size).range(2..=4).speed(0.05)), topic);
                        }
                        LayerSpec::Act(a) => {
                            let r = egui::ComboBox::from_id_salt(("act", i)).selected_text(RichText::new(a.name()).color(GREEN)).width(95.0).show_ui(ui, |ui| {
                                for k in ActKind::ALL {
                                    ui.selectable_value(a, k, k.name());
                                }
                            });
                            tp(t, r.response, topic);
                        }
                        LayerSpec::Dropout { p } => {
                            tlabel(ui, t, RichText::new("Dropout").strong(), topic);
                            tp(t, ui.add(egui::DragValue::new(p).range(0.0..=0.9).speed(0.01).fixed_decimals(2)), topic);
                        }
                        LayerSpec::BatchNorm => tlabel(ui, t, RichText::new("BatchNorm").strong().color(VIOLET), topic),
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
            tlabel(ui, t, RichText::new(format!("Ausgabe  Dense {} → Softmax ({}, fest)", self.n_classes(), fmt_n(*p))).monospace().color(MUTED), Topic::Output);
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Hinzufügen:");
            let l = &mut self.cfg.layers;
            for (name, spec) in [
                ("+ Dense", LayerSpec::Dense { units: 64 }),
                ("+ Aktivierung", LayerSpec::Act(ActKind::ReLU)),
                ("+ Conv", LayerSpec::Conv { filters: 8, k: 3, same: true }),
                ("+ MaxPool", LayerSpec::MaxPool { size: 2 }),
                ("+ BatchNorm", LayerSpec::BatchNorm),
                ("+ Dropout", LayerSpec::Dropout { p: 0.2 }),
            ] {
                if ui.small_button(name).clicked() {
                    *t = layer_topic(&spec);
                    l.push(spec);
                }
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

    fn cfg_init(&mut self, ui: &mut egui::Ui, t: &mut Topic) {
        let c = &mut self.cfg;
        combo(ui, "init", "Init:", &mut c.init, &Init::ALL, |i| i.name(), t, Topic::Init);
        combo(ui, "loss", "Verlust:", &mut c.loss, &[LossKind::CrossEntropy, LossKind::Mse], |l| l.name(), t, Topic::Loss);
        ui.horizontal(|ui| {
            tlabel(ui, t, "Label Smoothing ε:", Topic::LabelSmoothing);
            tp(t, ui.add(egui::Slider::new(&mut c.smoothing, 0.0..=0.3)), Topic::LabelSmoothing);
        });
    }

    fn cfg_opt(&mut self, ui: &mut egui::Ui, t: &mut Topic) {
        let c = &mut self.cfg;
        let ot = opt_topic(c.opt.kind);
        if combo(ui, "opt", "Verfahren:", &mut c.opt.kind, &OptKind::ALL, |o| o.name(), t, ot) {
            c.lr = c.opt.kind.default_lr();
            *t = opt_topic(c.opt.kind);
        }
        let row = |ui: &mut egui::Ui, t: &mut Topic, label: &str, topic: Topic, w: egui::Slider| {
            ui.horizontal(|ui| {
                tlabel(ui, t, label, topic);
                tp(t, ui.add(w), topic);
            });
        };
        row(ui, t, "Lernrate η:", Topic::LearningRate, egui::Slider::new(&mut c.lr, 1e-5..=2.0).logarithmic(true).custom_formatter(|v, _| format!("{v:.2e}")));
        match c.opt.kind {
            OptKind::Momentum | OptKind::Nesterov => row(ui, t, "Momentum μ:", Topic::Momentum, egui::Slider::new(&mut c.opt.momentum, 0.0..=0.99)),
            OptKind::RmsProp => {
                row(ui, t, "ρ (Abklingrate):", Topic::RmsProp, egui::Slider::new(&mut c.opt.beta2, 0.8..=0.9999));
                row(ui, t, "ε:", Topic::Epsilon, egui::Slider::new(&mut c.opt.eps, 1e-10..=1e-3).logarithmic(true).custom_formatter(|v, _| format!("{v:.0e}")));
            }
            OptKind::Adam | OptKind::AdamW => {
                row(ui, t, "β₁:", Topic::Betas, egui::Slider::new(&mut c.opt.beta1, 0.0..=0.99));
                row(ui, t, "β₂:", Topic::Betas, egui::Slider::new(&mut c.opt.beta2, 0.9..=0.9999));
                row(ui, t, "ε:", Topic::Epsilon, egui::Slider::new(&mut c.opt.eps, 1e-10..=1e-3).logarithmic(true).custom_formatter(|v, _| format!("{v:.0e}")));
            }
            OptKind::Sgd => {}
        }
        let wd_label = if c.opt.kind == OptKind::AdamW { "Weight Decay λ:" } else { "L2-Regularisierung λ:" };
        row(ui, t, wd_label, Topic::WeightDecay, egui::Slider::new(&mut c.opt.weight_decay, 0.0..=0.3).logarithmic(true).smallest_positive(1e-6).custom_formatter(|v, _| if v == 0.0 { "0".into() } else { format!("{v:.0e}") }));
        let mut clip = c.clip.is_some();
        ui.horizontal(|ui| {
            tp(t, ui.checkbox(&mut clip, "Gradient Clipping"), Topic::GradClip);
            let mut v = c.clip.unwrap_or(1.0);
            tp(t, ui.add_enabled(clip, egui::DragValue::new(&mut v).range(0.01..=100.0).speed(0.05).prefix("‖g‖ ≤ ")), Topic::GradClip);
            c.clip = clip.then_some(v);
        });
    }

    fn cfg_sched(&mut self, ui: &mut egui::Ui, t: &mut Topic) {
        let c = &mut self.cfg;
        combo(ui, "sched", "Plan:", &mut c.schedule, &Schedule::ALL, |s| s.name(), t, Topic::Schedule);
        match c.schedule {
            Schedule::Constant => {}
            Schedule::Step => {
                ui.horizontal(|ui| {
                    tlabel(ui, t, "alle", Topic::Schedule);
                    tp(t, ui.add(egui::DragValue::new(&mut c.step_every).range(1..=50)), Topic::Schedule);
                    tlabel(ui, t, "Epochen ×", Topic::Schedule);
                    tp(t, ui.add(egui::DragValue::new(&mut c.step_factor).range(0.01..=1.0).speed(0.01)), Topic::Schedule);
                });
            }
            Schedule::Exponential => {
                ui.horizontal(|ui| {
                    tlabel(ui, t, "γ pro Epoche:", Topic::Schedule);
                    tp(t, ui.add(egui::Slider::new(&mut c.exp_gamma, 0.5..=1.0)), Topic::Schedule);
                });
            }
            Schedule::Cosine => {
                ui.horizontal(|ui| {
                    tlabel(ui, t, "Minimum (Anteil):", Topic::Schedule);
                    tp(t, ui.add(egui::Slider::new(&mut c.lr_min_frac, 0.0..=0.5)), Topic::Schedule);
                });
            }
        }
        ui.horizontal(|ui| {
            tlabel(ui, t, "Warmup:", Topic::Warmup);
            tp(t, ui.add(egui::Slider::new(&mut c.warmup_epochs, 0.0..=5.0).suffix(" Ep.")), Topic::Warmup);
        });
        let pts: Vec<[f64; 2]> = (0..=200)
            .map(|i| {
                let e = i as f32 / 200.0 * c.epochs as f32;
                [e as f64, c.lr_at(e) as f64]
            })
            .collect();
        let mx = pts.iter().map(|p| p[1]).fold(0.0, f64::max);
        let r = Plot::new("lrplan").height(90.0).allow_drag(false).allow_zoom(false).allow_scroll(false).include_y(0.0).include_y(mx * 1.15).x_axis_label("Epoche").show(ui, |p| {
            p.line(Line::new("η", PlotPoints::from(pts)).color(ACCENT));
        });
        tp(t, r.response, Topic::Schedule);
    }

    fn cfg_train(&mut self, ui: &mut egui::Ui, t: &mut Topic) {
        let cores = self.cores;
        let c = &mut self.cfg;
        let row = |ui: &mut egui::Ui, t: &mut Topic, label: &str, topic: Topic, w: egui::Slider| {
            ui.horizontal(|ui| {
                tlabel(ui, t, label, topic);
                tp(t, ui.add(w), topic);
            });
        };
        row(ui, t, "Epochen:", Topic::Epochs, egui::Slider::new(&mut c.epochs, 1..=100));
        row(ui, t, "Batchgröße:", Topic::BatchSize, egui::Slider::new(&mut c.batch, 1..=1024).logarithmic(true));
        c.threads = c.threads.clamp(1, cores);
        row(ui, t, &format!("CPU-Threads (von {cores}):"), Topic::Threads, egui::Slider::new(&mut c.threads, 1..=cores));
        ui.horizontal(|ui| {
            tlabel(ui, t, "Seed:", Topic::Seed);
            tp(t, ui.add(egui::DragValue::new(&mut c.seed)), Topic::Seed);
            if ui.small_button("🎲").clicked() {
                c.seed = (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(1) % 1_000_000) as u64;
                *t = Topic::Seed;
            }
        });
        ui.horizontal(|ui| {
            tlabel(ui, t, "Test-Stichprobe alle", Topic::EvalEvery);
            tp(t, ui.add(egui::DragValue::new(&mut c.eval_every).range(0..=2000).speed(5)), Topic::EvalEvery);
            tlabel(ui, t, "Schritte", Topic::EvalEvery);
        });
    }
}

// ---------------------------------------------------------------------------
// Tutorial-Leiste
// ---------------------------------------------------------------------------

impl App {
    fn tutorial_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("🎓 Tutorial").size(17.0).strong().color(ACCENT));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("✕").on_hover_text("Tutorial ausblenden").clicked() {
                    self.show_tutorial = false;
                }
            });
        });
        ui.label(RichText::new("Klicke auf einen Parameter, einen Tab oder ein Diagramm.").small().color(MUTED));
        ui.separator();
        let d = tutorial::doc(self.topic);
        ui.label(RichText::new(d.title).size(19.0).strong());
        ui.add_space(4.0);
        if let Some(c) = tutorial::current(self.topic, &self.cfg, self.cores) {
            egui::Frame::new().fill(Color32::from_rgb(40, 34, 24)).corner_radius(4.0).inner_margin(6.0).show(ui, |ui| {
                ui.label(RichText::new(c).color(ACCENT));
            });
            ui.add_space(4.0);
        }
        ui.label(RichText::new(d.what).size(14.0));
        if !d.math.is_empty() {
            ui.add_space(6.0);
            ui.label(RichText::new("Mathematik").strong().color(BLUE));
            egui::Frame::new().fill(Color32::from_gray(22)).corner_radius(4.0).inner_margin(8.0).show(ui, |ui| {
                ui.label(RichText::new(d.math).monospace().size(13.0));
            });
        }
        ui.add_space(6.0);
        ui.label(RichText::new("Übliche Werte").strong().color(GREEN));
        ui.label(d.typical);
        ui.add_space(6.0);
        ui.label(RichText::new("Beobachten & ausprobieren").strong().color(VIOLET));
        ui.label(d.watch);
        if !d.related.is_empty() {
            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Siehe auch:").color(MUTED));
                for &r in d.related {
                    if ui.link(tutorial::doc(r).title).clicked() {
                        self.topic = r;
                    }
                }
            });
        }
        ui.add_space(10.0);
        ui.separator();
        egui::CollapsingHeader::new(RichText::new("Alle Themen").strong()).default_open(false).show(ui, |ui| {
            for (group, topics) in tutorial::index() {
                ui.label(RichText::new(group).color(MUTED).small());
                ui.horizontal_wrapped(|ui| {
                    for tt in topics {
                        if ui.selectable_label(self.topic == tt, tutorial::doc(tt).title).clicked() {
                            self.topic = tt;
                        }
                    }
                });
            }
        });
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
fn chart(ui: &mut egui::Ui, slot: &mut Topic, topic: Topic, id: &str, title: &str, height: f32, log: bool, series: Vec<Series>, hline: Option<(f64, &str)>, x_label: &str) {
    tlabel(ui, slot, RichText::new(title).strong(), topic);
    let mut p = Plot::new(id).height(height.max(80.0)).legend(Legend::default()).x_axis_label(x_label).allow_scroll(false);
    if log {
        p = p.y_axis_formatter(|m, _| viz::log_label(m.value));
    }
    let r = p.show(ui, |pu| {
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
    tp(slot, r.response, topic);
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap_or(0)
}

/// Wahrscheinlichkeitsbalken; bei vielen Klassen nur die zehn wahrscheinlichsten.
fn bar_probs(ui: &mut egui::Ui, probs: &[f32], names: &[String], truth: Option<usize>, width: f32) {
    let pred = argmax(probs);
    let mut order: Vec<usize> = (0..probs.len()).collect();
    if probs.len() > 12 {
        order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        order.truncate(10);
        ui.label(RichText::new(format!("die 10 wahrscheinlichsten von {} Klassen", probs.len())).small().color(MUTED));
    }
    for i in order {
        let p = probs[i];
        ui.horizontal(|ui| {
            let short: String = names.get(i).map(|s| s.chars().take(14).collect()).unwrap_or_default();
            let mut name = RichText::new(format!("{short:>14}")).monospace();
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

fn short(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
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
                ui.label("Links eine Voreinstellung oder eigene Parameter wählen, dann oben „Training starten“.");
                ui.label(RichText::new("Tipp: Der Tab „Netz & Aktivierungen“ zeigt das untrainierte Netz schon jetzt.").color(MUTED));
            });
            return;
        }
        let h = (ui.available_height() - 70.0) / 2.0 - 26.0;
        let topic = &mut self.topic;
        ui.columns(2, |c| {
            chart(
                &mut c[0],
                topic,
                Topic::ViewLoss,
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
            chart(&mut c[1], topic, Topic::ViewAcc, "acc", "Genauigkeit [%]", h, false, vec![line("Train-Batches (geglättet)", acc_s, ACCENT), Series { name: "Test".into(), pts: val_acc, color: BLUE, width: 2.0, dots: true }], None, "Schritt");
        });
        ui.columns(3, |c| {
            let lr_max = lr.iter().map(|p| p[1]).fold(0.0, f64::max);
            tlabel(&mut c[0], topic, RichText::new("Lernrate η").strong(), Topic::Schedule);
            let r = Plot::new("lr").height(h.max(80.0)).include_y(0.0).include_y(lr_max * 1.15).x_axis_label("Schritt").allow_scroll(false).show(&mut c[0], |p| {
                p.line(Line::new("η", PlotPoints::from(lr)).color(GREEN).width(1.6));
            });
            tp(topic, r.response, Topic::Schedule);
            chart(&mut c[1], topic, Topic::GradClip, "gn", "Globale Gradientennorm ‖∇θ‖ (log)", h, true, vec![line("‖∇θ‖", gn, Color32::from_rgb(230, 90, 160))], None, "Schritt");
            let se = vec![Series { name: "Train-Acc".into(), pts: ep_tr, color: ACCENT, width: 2.0, dots: true }, Series { name: "Test-Acc".into(), pts: ep_te, color: BLUE, width: 2.0, dots: true }];
            chart(&mut c[2], topic, Topic::ViewAcc, "ep", "Pro Epoche: Train vs. Test [%]", h, false, se, None, "Epoche");
        });
    }
}

// ---------------------------------------------------------------------------
// Tab: Netz & Aktivierungen
// ---------------------------------------------------------------------------

impl App {
    fn probe_selector(&mut self, ui: &mut egui::Ui) {
        let n = self.dataset().map(|d| d.n_test()).unwrap_or(10_000).max(1);
        let wrong: Vec<usize> = self.tr.shared.lock().unwrap().last_eval.as_ref().map(|e| e.wrong.iter().map(|w| w.0).collect()).unwrap_or_default();
        ui.horizontal_wrapped(|ui| {
            let mut idx = match self.probe {
                Probe::Test(i) => i.min(n - 1),
                Probe::Drawing => 0,
            };
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
        let ctx = ui.ctx().clone();
        let probe = self.probe;
        let (key, img, probs, outs, input, model, cfg, trained, ds) = match self.trace() {
            Ok((tr, model, cfg, trained, ds)) => (tr.key, tr.img.clone(), tr.probs.clone(), tr.outs.clone(), tr.input.clone(), model, cfg, trained, ds),
            Err(e) => {
                ui.add_space(30.0);
                ui.label(RichText::new(e).size(16.0));
                return;
            }
        };
        let names = &ds.classes;
        let truth = match probe {
            Probe::Test(i) => Some(ds.test_y[i.min(ds.n_test() - 1)] as usize),
            _ => None,
        };
        if !trained {
            ui.label(RichText::new("Untrainiertes Netz (zufällige Startgewichte). Starte das Training und sieh zu, wie sich die Aktivierungen verändern.").color(ACCENT));
        }
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(310.0);
                ui.label(RichText::new("Eingabe").strong());
                let side = 196.0;
                let t = self.tex.get(&ctx, "probe_img", key, || if probe == Probe::Drawing { viz::sample(&img, 1, 28, 28) } else { viz::sample(&img, ds.c, ds.h, ds.w) });
                image(ui, &t, side, side * ds.h as f32 / ds.w as f32);
                let ti = self.tex.get(&ctx, "probe_in", key, || act_image(&input, model.input));
                ui.horizontal(|ui| {
                    let sz = ti.size_vec2();
                    let s = 84.0 / sz.y;
                    image(ui, &ti, sz.x * s, 84.0);
                    let mn = input.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mx = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    tlabel(ui, &mut self.topic, RichText::new(format!("ins Netz (normiert)\n{}\nmin {mn:+.2}\nmax {mx:+.2}", cfg.norm.name())).small().color(MUTED), Topic::Normalization);
                });
                if let Some(t) = truth {
                    ui.label(format!("Wahres Label: {}", names[t]));
                }
                let pred = argmax(&probs);
                ui.label(RichText::new(format!("Vorhersage: {}  ({:.1} %)", names.get(pred).map(|s| s.as_str()).unwrap_or("?"), probs[pred] * 100.0)).size(17.0).strong().color(if truth.is_none_or(|t| t == pred) { GREEN } else { RED }));
                ui.add_space(6.0);
                tlabel(ui, &mut self.topic, RichText::new("Softmax-Ausgabe").strong(), Topic::Output);
                bar_probs(ui, &probs, names, truth, 110.0);
            });
            ui.separator();
            egui::ScrollArea::vertical().id_salt("layers").auto_shrink(false).show(ui, |ui| {
                ui.vertical(|ui| {
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
                                    tlabel(ui, &mut self.topic, RichText::new(format!("{}  {title}", li + 1)).strong().color(layer_color(l)), layer_topic_of(l, li + 1 == nl));
                                    ui.label(RichText::new(format!("{} → {}", shape.describe(), os.describe())).monospace().small());
                                    if l.n_params() > 0 {
                                        ui.label(RichText::new(format!("{} Parameter, {} MACs", fmt_n(l.n_params()), fmt_n(l.macs()))).small().color(MUTED));
                                    }
                                    let n = out.len().max(1) as f32;
                                    let mean = out.iter().sum::<f32>() / n;
                                    let sd = (out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
                                    let mut info = format!("μ={mean:+.3}  σ={sd:.3}");
                                    if let Layer::Act(a) = l {
                                        if a.kind == ActKind::ReLU {
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
                });
            });
        });
    }
}

fn layer_topic_of(l: &Layer, last: bool) -> Topic {
    if last {
        return Topic::Output;
    }
    match l {
        Layer::Dense(_) => Topic::Dense,
        Layer::Conv(_) => Topic::Conv,
        Layer::Pool(_) => Topic::MaxPool,
        Layer::Act(_) => Topic::Activation,
        Layer::Dropout(_) => Topic::Dropout,
        Layer::BatchNorm(_) => Topic::BatchNorm,
    }
}

fn layer_color(l: &Layer) -> Color32 {
    match l {
        Layer::Dense(_) => ACCENT,
        Layer::Conv(_) => Color32::from_rgb(230, 90, 160),
        Layer::Pool(_) => Color32::from_rgb(60, 210, 200),
        Layer::Act(_) => GREEN,
        Layer::Dropout(_) => MUTED,
        Layer::BatchNorm(_) => VIOLET,
    }
}

/// Aktivierungsbild: räumlich = eine Kachel pro Kanal, sonst Pixelraster.
fn act_image(out: &[f32], s: Shape) -> ColorImage {
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

#[allow(clippy::too_many_arguments)]
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
        let first_param = pl.first() == Some(&li);
        let inp = model.input;
        let layer = &model.layers[li];
        let ps = layer.params();
        let (w, g) = (&ps[0].w, &ps[0].g);
        let avail = ui.available_width() - 20.0;
        let top_h = (ui.available_height() * 0.58).max(200.0);
        let v = ver.wrapping_mul(131) ^ li as u64;
        egui::ScrollArea::vertical().id_salt("wts").max_height(top_h).auto_shrink([false, true]).show(ui, |ui| match layer {
            Layer::Dense(d) if first_param && d.inp == inp.len() && inp.spatial() => {
                let (h, ww, c) = (inp.h, inp.w, inp.c);
                ui.label(RichText::new(format!("Eingehende Gewichte jedes der {} Neuronen als {h}×{ww}-Bild: das Muster, auf das ein Neuron anspricht{}.", d.out, if c == 3 { " (Farbe = Gewicht pro Kanal, Grau = 0)" } else { "" })).color(MUTED));
                let n = d.out.min(256);
                let cols = if n <= 10 { n } else { ((avail / 64.0) as usize).clamp(4, 32) };
                let il = d.inp;
                let hw = h * ww;
                let img = if c == 3 {
                    viz::rgb_tiles(n, ww, h, cols, |t| max_abs(&w[t * il..][..il]), |t, ch, x, y| w[t * il + ch * hw + y * ww + x])
                } else {
                    viz::tiles(n, ww, h, cols, |t| viz::Scale::Signed(max_abs(&w[t * il..][..il])), |t, x, y| w[t * il + y * ww + x])
                };
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
            Layer::Conv(c) if c.cin == 3 => {
                ui.label(RichText::new(format!("{} Farbfilter à {}×{}: Farbe = Gewicht pro Kanal (R, G, B), Grau = 0.", c.cout, c.k, c.k)).color(MUTED));
                let (kk, ckk) = (c.k * c.k, 3 * c.k * c.k);
                let img = viz::rgb_tiles(c.cout, c.k, c.k, c.cout.min(16), |t| max_abs(&w[t * ckk..][..ckk]), |t, ch, x, y| w[t * ckk + ch * kk + y * c.k + x]);
                show_img(ui, &mut self.tex, &ctx, "w_conv", v, img, avail.min(16.0 * 110.0), 24.0);
            }
            Layer::Conv(c) => {
                let cin = c.cin.min(16);
                ui.label(RichText::new(format!("{} Filter mit je {}×{}×{} Gewichten. Zeile = Ausgabekanal, Spalte = Eingangskanal{}.", c.cout, c.cin, c.k, c.k, if c.cin > 16 { " (erste 16)" } else { "" })).color(MUTED));
                let kk = c.k * c.k;
                let ckk = c.cin * kk;
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
        let topic = &mut self.topic;
        ui.columns(2, |c| {
            tlabel(&mut c[0], topic, RichText::new("Histogramm der Gewichte").strong(), Topic::ViewHistogram);
            c[0].label(RichText::new(stats(w)).small().monospace().color(MUTED));
            let r = Plot::new("hw").height(h).show(&mut c[0], |p| p.bar_chart(BarChart::new("W", hw).color(ACCENT)));
            tp(topic, r.response, Topic::ViewHistogram);
            tlabel(&mut c[1], topic, RichText::new("Histogramm des letzten Gradienten ∂L/∂W").strong(), Topic::ViewHistogram);
            c[1].label(RichText::new(stats(g)).small().monospace().color(MUTED));
            if g.iter().all(|&x| x == 0.0) {
                c[1].add_space(30.0);
                c[1].label(RichText::new("Noch kein Gradient: Er entsteht beim ersten Trainingsschritt (z. B. mit „Schrittweise“ und ⏭ 1).").color(MUTED));
            } else {
                let r = Plot::new("hg").height(h).show(&mut c[1], |p| p.bar_chart(BarChart::new("∂L/∂W", hg).color(BLUE)));
                tp(topic, r.response, Topic::ViewHistogram);
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
                ui.label("Gut zu beobachten mit dem Lehrbeispiel „Vanishing Gradients“: Die Gradienten der vorderen Schichten sind um Größenordnungen kleiner.");
            });
            return;
        }
        let h = (ui.available_height() - 80.0) / 2.0 - 26.0;
        let topic = &mut self.topic;
        ui.columns(2, |c| {
            chart(&mut c[0], topic, Topic::ViewGradNorm, "gnl", "‖∂L/∂W‖ pro Schicht (log)", h, true, gn, None, "Schritt");
            chart(&mut c[1], topic, Topic::ViewUpdateRatio, "url", "Update-Verhältnis ‖ΔW‖ / ‖W‖ pro Schritt (log)", h, true, ur, Some((-3.0, "Faustregel ≈ 1e-3")), "Schritt");
        });
        ui.columns(2, |c| {
            chart(&mut c[0], topic, Topic::ViewActStats, "astd", "Standardabweichung der Aktivierungen (Trainingsbatch)", h, false, astd, None, "Schritt");
            chart(&mut c[1], topic, Topic::ViewActStats, "dead", "Inaktive (ReLU ≤ 0) bzw. gesättigte (Sigmoid, Tanh) Einheiten [%]", h, false, dead, None, "Schritt");
        });
    }
}

// ---------------------------------------------------------------------------
// Tab: Auswertung
// ---------------------------------------------------------------------------

impl App {
    fn tab_auswertung(&mut self, ui: &mut egui::Ui) {
        let (ev, spec) = {
            let s = self.tr.shared.lock().unwrap();
            (s.last_eval.clone(), s.cfg.dataset.clone())
        };
        let Some(ev) = ev else {
            ui.add_space(40.0);
            ui.vertical_centered(|ui| ui.label(RichText::new("Die Auswertung auf dem ganzen Testset erscheint nach der ersten Epoche.").size(18.0)));
            return;
        };
        let Some(ds) = self.dataset().filter(|d| d.spec == spec) else {
            ui.label(format!("Die Auswertung gehört zu „{}“, das gerade nicht geladen ist.", spec.label()));
            return;
        };
        let names = &ds.classes;
        let k = ev.confusion.len();
        let ctx = ui.ctx().clone();
        let errors: u32 = ev.confusion.iter().enumerate().map(|(i, r)| r.iter().sum::<u32>() - r[i]).sum();
        ui.label(RichText::new(format!("Testset: Genauigkeit {:.2} %  ·  Verlust {:.4}  ·  {errors} Fehler", ev.acc * 100.0, ev.loss)).size(16.0).strong());
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                tlabel(ui, &mut self.topic, RichText::new("Konfusionsmatrix (Zeile = wahr, Spalte = vorhergesagt)").strong(), Topic::ViewConfusion);
                let cell = (440.0 / k as f32).clamp(14.0, 42.0);
                let lab = 80.0;
                let (rect, resp) = ui.allocate_exact_size(egui::vec2(lab + cell * k as f32 + 70.0, 24.0 + cell * k as f32 + 4.0), Sense::click());
                tp(&mut self.topic, resp, Topic::ViewConfusion);
                let p = ui.painter_at(rect);
                let o = rect.min + egui::vec2(lab, 24.0);
                let maxoff = ev.confusion.iter().enumerate().flat_map(|(i, r)| r.iter().enumerate().filter(move |(j, _)| *j != i).map(|(_, &v)| v)).max().unwrap_or(1).max(1);
                let fs = (cell * 0.28).clamp(8.0, 11.0);
                for i in 0..k {
                    p.text(o + egui::vec2(-6.0, (i as f32 + 0.5) * cell), egui::Align2::RIGHT_CENTER, short(&names[i], 11), egui::FontId::proportional(fs + 1.0), MUTED);
                    if cell >= 24.0 {
                        p.text(o + egui::vec2((i as f32 + 0.5) * cell, -4.0), egui::Align2::CENTER_BOTTOM, short(&names[i], 4), egui::FontId::proportional(fs), MUTED);
                    }
                    let row: u32 = ev.confusion[i].iter().sum();
                    for j in 0..k {
                        let v = ev.confusion[i][j];
                        let r = egui::Rect::from_min_size(o + egui::vec2(j as f32 * cell, i as f32 * cell), egui::vec2(cell - 1.5, cell - 1.5));
                        let col = if i == j {
                            let t = v as f32 / row.max(1) as f32;
                            Color32::from_rgb((20.0 + 40.0 * t) as u8, (40.0 + 130.0 * t) as u8, (30.0 + 40.0 * t) as u8)
                        } else if v == 0 {
                            Color32::from_gray(28)
                        } else {
                            let t = ((v as f32).ln_1p() / (maxoff as f32).ln_1p()).clamp(0.15, 1.0);
                            Color32::from_rgb((60.0 + 180.0 * t) as u8, (30.0 + 40.0 * t) as u8, (30.0 + 30.0 * t) as u8)
                        };
                        p.rect_filled(r, 2.0, col);
                        if v > 0 && cell >= 22.0 {
                            p.text(r.center(), egui::Align2::CENTER_CENTER, v.to_string(), egui::FontId::monospace(fs), Color32::WHITE);
                        }
                    }
                    let rec = ev.confusion[i][i] as f32 / row.max(1) as f32;
                    p.text(o + egui::vec2(k as f32 * cell + 6.0, (i as f32 + 0.5) * cell), egui::Align2::LEFT_CENTER, format!("{:.1} %", rec * 100.0), egui::FontId::monospace(fs), if rec > 0.98 { GREEN } else { ACCENT });
                }
                ui.label(RichText::new("Rechts daneben: Recall (Trefferquote) pro Klasse.").small().color(MUTED));
            });
            ui.separator();
            ui.vertical(|ui| {
                tlabel(ui, &mut self.topic, RichText::new(format!("Fehlklassifikationen: die {} überzeugtesten Irrtümer. Klick = im Netz ansehen.", ev.wrong.len().min(120))).strong(), Topic::ViewErrors);
                egui::ScrollArea::vertical().id_salt("wrong").auto_shrink(false).show(ui, |ui| {
                    let cols = ((ui.available_width() / 96.0) as usize).max(1);
                    egui::Grid::new("wg").spacing(egui::vec2(6.0, 6.0)).show(ui, |ui| {
                        for (n, &(idx, pred, truth, conf)) in ev.wrong.iter().take(120).enumerate() {
                            ui.vertical(|ui| {
                                let mut hh = std::collections::hash_map::DefaultHasher::new();
                                (idx, &ds.spec).hash(&mut hh);
                                let t = self.tex.get(&ctx, &format!("wrong{n}"), hh.finish(), || viz::sample(ds.test_image(idx), ds.c, ds.h, ds.w));
                                let r = ui.add(egui::Image::new(egui::load::SizedTexture::new(t.id(), egui::vec2(84.0, 84.0))).sense(Sense::click()));
                                if r.on_hover_text(format!("Testbild {idx}: {} als {} erkannt", names[truth as usize], names[pred as usize])).clicked() {
                                    self.probe = Probe::Test(idx);
                                    self.tab = Tab::Netz;
                                }
                                ui.label(RichText::new(format!("{}→{} {:.0}%", short(&names[truth as usize], 5), short(&names[pred as usize], 5), conf * 100.0)).small().monospace());
                            });
                            if (n + 1) % cols == 0 {
                                ui.end_row();
                            }
                        }
                    });
                });
            });
        });
    }
}

// ---------------------------------------------------------------------------
// Tab: Zeichnen
// ---------------------------------------------------------------------------

impl App {
    fn tab_zeichnen(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        if self.cfg.dataset.shape() != (1, 28, 28) && self.model().is_some_and(|m| m.0.input != Shape::new(1, 28, 28)) {
            ui.add_space(30.0);
            ui.label(RichText::new("Zeichnen funktioniert nur mit Datensätzen aus 1×28×28-Graustufenbildern (MNIST, Fashion-MNIST, KMNIST, EMNIST).").size(16.0));
            return;
        }
        ui.label(RichText::new("Male eine Ziffer mit der Maus. Sie wird wie MNIST aufbereitet (in 20×20 eingepasst, Schwerpunkt zentriert) und live klassifiziert.").color(MUTED));
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                let size = 420.0;
                let (resp, painter) = ui.allocate_painter(egui::vec2(size, size), Sense::drag());
                let rect = resp.rect;
                if resp.dragged() || resp.drag_started() {
                    self.topic = Topic::ViewDrawing;
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
                let res = self.trace().map(|(t, _, cfg, trained, ds)| (t.img.clone(), t.probs.clone(), cfg.dataset.kind(), trained, t.key, ds));
                self.probe = prev;
                let (img, probs, kind, trained, key, ds) = match res {
                    Ok(r) => r,
                    Err(e) => {
                        ui.label(e);
                        return;
                    }
                };
                ui.label(RichText::new("Aufbereitet (28×28)").strong());
                let t = self.tex.get(&ctx, "draw28", key, || viz::sample(&img, 1, 28, 28));
                image(ui, &t, 168.0, 168.0);
                if !trained {
                    ui.label(RichText::new("Das Netz ist noch untrainiert.").color(ACCENT));
                }
                if kind != data::DataKind::Digits {
                    ui.label(RichText::new(format!("Hinweis: Das Netz kennt die Klassen von „{}“.", ds.name)).color(ACCENT));
                }
                let pred = argmax(&probs);
                ui.label(RichText::new(format!("Vorhersage: {}", ds.classes.get(pred).map(|s| s.as_str()).unwrap_or("?"))).size(26.0).strong().color(GREEN));
                bar_probs(ui, &probs, &ds.classes, None, 220.0);
            });
        });
    }
}

// ---------------------------------------------------------------------------
// Tab: Berater
// ---------------------------------------------------------------------------

impl App {
    fn advice_list(&mut self, ui: &mut egui::Ui, list: Vec<Advice>, id: &str) {
        let mut apply: Option<Box<dyn Fn(&mut TrainConfig)>> = None;
        let mut label = String::new();
        for (n, a) in list.into_iter().enumerate() {
            let (icon, col) = match a.level {
                Level::Error => ("⛔", RED),
                Level::Warn => ("⚠", ACCENT),
                Level::Tip => ("💡", BLUE),
                Level::Good => ("✔", GREEN),
            };
            egui::Frame::group(ui.style()).inner_margin(8.0).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(RichText::new(icon).size(16.0).color(col));
                    ui.label(RichText::new(&a.title).strong().color(col).size(15.0));
                });
                ui.label(&a.text);
                ui.horizontal(|ui| {
                    if let Some(f) = a.fix {
                        let running = self.tr.running();
                        if ui.add_enabled(!running, egui::Button::new(RichText::new(format!("✔ {}", f.label)).color(GREEN))).on_hover_text("Übernimmt die Korrektur in die Konfiguration (gilt ab dem nächsten Training).").clicked() {
                            label = f.label.clone();
                            apply = Some(f.apply);
                        }
                    }
                    if ui.push_id((id, n), |ui| ui.button("🎓 Erklärung")).inner.clicked() {
                        self.topic = a.topic;
                        self.show_tutorial = true;
                    }
                });
            });
        }
        if let Some(f) = apply {
            f(&mut self.cfg);
            self.info_line = Some(format!("Berater: „{label}“ übernommen."));
        }
    }

    fn tab_berater(&mut self, ui: &mut egui::Ui) {
        let info = self.data_info();
        let running = self.tr.running();
        egui::ScrollArea::vertical().id_salt("berater").auto_shrink(false).show(ui, |ui| {
            // Empfehlung für den Datensatz
            let (recipe, why) = advisor::ideal(&info);
            egui::Frame::new().fill(Color32::from_rgb(28, 36, 30)).corner_radius(6.0).inner_margin(12.0).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new(format!("Empfehlung für {}", self.cfg.dataset.label())).size(17.0).strong().color(GREEN));
                ui.label(why);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Passende Voreinstellung:").color(MUTED));
                    ui.label(RichText::new(recipe).strong());
                    if ui.add_enabled(!running, egui::Button::new(RichText::new("✔ Übernehmen").color(GREEN))).clicked() {
                        self.apply_recipe(recipe);
                    }
                });
            });
            ui.add_space(10.0);
            ui.label(RichText::new("Aktuelle Konfiguration").size(16.0).strong());
            let list = advisor::check_config(&self.cfg, &info, &self.preview, self.cores);
            self.advice_list(ui, list, "cfg");

            let (hist_len, list) = {
                let s = self.tr.shared.lock().unwrap();
                let k = s.model.as_ref().map(|m| m.classes).unwrap_or(10);
                (s.hist.loss.len(), advisor::check_training(&s.cfg, &s.hist, &s.status, &s.layer_names, &s.param_layers, k, s.cfg.dataset.kind()))
            };
            ui.add_space(10.0);
            ui.label(RichText::new("Letzter Trainingslauf").size(16.0).strong());
            if hist_len < 30 && list.is_empty() {
                ui.label(RichText::new("Noch kein (ausreichend langer) Lauf. Nach dem Training erkennt der Berater hier Overfitting, Underfitting, verschwindende Gradienten, tote ReLUs und eine falsch gewählte Lernrate.").color(MUTED));
            } else if list.is_empty() {
                ui.label(RichText::new("Keine Auffälligkeiten im Verlauf.").color(GREEN));
            } else {
                self.advice_list(ui, list, "train");
            }
        });
    }
}
