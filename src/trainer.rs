//! Training im Hintergrund-Thread. Die Oberfläche liest den Zustand über
//! `Shared` und steuert über die Atomics in `Control`.

use crate::data::{self, Augment, DataSpec, Dataset, Norm};
use crate::nn::{self, ActStat, Init, LayerSpec, LossKind, Model, OptCfg, OptKind, Shape};
use crate::rng::Rng;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    Constant,
    Step,
    Exponential,
    Cosine,
}

impl Schedule {
    pub const ALL: [Schedule; 4] = [Schedule::Constant, Schedule::Step, Schedule::Exponential, Schedule::Cosine];
    pub fn name(self) -> &'static str {
        match self {
            Schedule::Constant => "Konstant",
            Schedule::Step => "Stufen (Step Decay)",
            Schedule::Exponential => "Exponentiell",
            Schedule::Cosine => "Cosine Annealing",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub dataset: DataSpec,
    pub train_size: usize,
    pub norm: Norm,
    pub aug: Augment,
    pub layers: Vec<LayerSpec>,
    pub init: Init,
    pub loss: LossKind,
    pub smoothing: f32,
    pub opt: OptCfg,
    pub lr: f32,
    pub schedule: Schedule,
    pub step_every: usize,
    pub step_factor: f32,
    pub exp_gamma: f32,
    pub lr_min_frac: f32,
    pub warmup_epochs: f32,
    pub epochs: usize,
    pub batch: usize,
    pub clip: Option<f32>,
    pub threads: usize,
    pub seed: u64,
    pub eval_every: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            dataset: DataSpec::mnist(),
            train_size: 60_000,
            norm: Norm::Standard,
            aug: Augment::default(),
            layers: vec![LayerSpec::Dense { units: 128 }, LayerSpec::Act(nn::ActKind::ReLU)],
            init: Init::He,
            loss: LossKind::CrossEntropy,
            smoothing: 0.0,
            opt: OptCfg { kind: OptKind::Adam, momentum: 0.9, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0 },
            lr: 0.001,
            schedule: Schedule::Constant,
            step_every: 3,
            step_factor: 0.3,
            exp_gamma: 0.9,
            lr_min_frac: 0.01,
            warmup_epochs: 0.0,
            epochs: 10,
            batch: 64,
            clip: None,
            threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
            seed: 42,
            eval_every: 100,
        }
    }
}

impl TrainConfig {
    /// Lernrate zum (gebrochenen) Epochenzeitpunkt `e`.
    pub fn lr_at(&self, e: f32) -> f32 {
        let base = self.lr;
        let total = self.epochs.max(1) as f32;
        let mut lr = match self.schedule {
            Schedule::Constant => base,
            Schedule::Step => base * self.step_factor.powi((e / self.step_every.max(1) as f32).floor() as i32),
            Schedule::Exponential => base * self.exp_gamma.powf(e),
            Schedule::Cosine => {
                let lo = base * self.lr_min_frac;
                let t = ((e - self.warmup_epochs) / (total - self.warmup_epochs).max(1e-3)).clamp(0.0, 1.0);
                lo + 0.5 * (base - lo) * (1.0 + (std::f32::consts::PI * t).cos())
            }
        };
        if self.warmup_epochs > 0.0 && e < self.warmup_epochs {
            lr *= (e / self.warmup_epochs).max(0.01);
        }
        lr
    }

    pub fn input_shape(&self) -> Shape {
        let (c, h, w) = self.dataset.shape();
        Shape::new(c, h, w)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Idle,
    Loading(String),
    Running,
    Paused,
    Finished,
    Stopped,
    Error(String),
}

/// Ergebnis einer Auswertung auf dem Testset.
#[derive(Clone, Default)]
pub struct Eval {
    pub loss: f32,
    pub acc: f32,
    pub confusion: Vec<Vec<u32>>,
    /// (Testindex, Vorhersage, Wahrheit, Konfidenz der Vorhersage)
    pub wrong: Vec<(usize, u8, u8, f32)>,
}

#[derive(Default)]
pub struct History {
    /// x = Schritt
    pub loss: Vec<[f64; 2]>,
    pub acc: Vec<[f64; 2]>,
    pub lr: Vec<[f64; 2]>,
    pub grad_global: Vec<[f64; 2]>,
    pub val_loss: Vec<[f64; 2]>,
    pub val_acc: Vec<[f64; 2]>,
    /// Pro Epoche: Mittelwerte auf den Trainingsbatches
    pub epoch_train_loss: Vec<[f64; 2]>,
    pub epoch_train_acc: Vec<[f64; 2]>,
    pub epoch_val_loss: Vec<[f64; 2]>,
    pub epoch_val_acc: Vec<[f64; 2]>,
    /// Pro Schicht mit Gewichten: Verlauf von ‖∇W‖ und ‖ΔW‖/‖W‖
    pub grad_norm: Vec<Vec<[f64; 2]>>,
    pub upd_ratio: Vec<Vec<[f64; 2]>>,
    /// Pro Schicht: Std. der Ausgabe und Anteil toter/gesättigter Einheiten
    pub act_std: Vec<Vec<[f64; 2]>>,
    pub act_dead: Vec<Vec<[f64; 2]>>,
}

pub struct Shared {
    pub status: Status,
    pub cfg: TrainConfig,
    pub epoch: usize,
    pub step: u64,
    pub steps_per_epoch: usize,
    pub samples_per_sec: f64,
    pub gflops: f64,
    pub elapsed: Duration,
    pub hist: History,
    /// Schichtnamen und Indizes der Schichten mit Gewichten
    pub layer_names: Vec<String>,
    pub param_layers: Vec<usize>,
    pub model: Option<Arc<Model>>,
    pub model_version: u64,
    pub last_eval: Option<Eval>,
    pub macs: usize,
    pub n_params: usize,
}

pub struct Control {
    pub pause: AtomicBool,
    pub stop: AtomicBool,
    pub steps: AtomicU32,
    /// Lernraten-Faktor, live änderbar (×1000)
    pub lr_scale_milli: AtomicU32,
    /// Zeitlupe: Pause nach jedem Schritt in ms
    pub delay_ms: AtomicU32,
}

pub struct Trainer {
    pub shared: Arc<Mutex<Shared>>,
    pub ctl: Arc<Control>,
    pub data: Arc<Mutex<Option<Arc<Dataset>>>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Trainer {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(Shared {
                status: Status::Idle,
                cfg: TrainConfig::default(),
                epoch: 0,
                step: 0,
                steps_per_epoch: 0,
                samples_per_sec: 0.0,
                gflops: 0.0,
                elapsed: Duration::ZERO,
                hist: History::default(),
                layer_names: Vec::new(),
                param_layers: Vec::new(),
                model: None,
                model_version: 0,
                last_eval: None,
                macs: 0,
                n_params: 0,
            })),
            ctl: Arc::new(Control { pause: AtomicBool::new(false), stop: AtomicBool::new(false), steps: AtomicU32::new(0), lr_scale_milli: AtomicU32::new(1000), delay_ms: AtomicU32::new(0) }),
            data: Arc::new(Mutex::new(None)),
            handle: None,
        }
    }

    pub fn running(&self) -> bool {
        self.handle.as_ref().is_some_and(|h| !h.is_finished())
    }

    pub fn stop(&mut self) {
        self.ctl.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.ctl.stop.store(false, Ordering::SeqCst);
    }

    /// Startet einen neuen Lauf. `paused`: mit Einzelschritten beginnen.
    pub fn start(&mut self, cfg: TrainConfig, paused: bool, repaint: impl Fn() + Send + Sync + 'static) {
        self.stop();
        self.ctl.pause.store(paused, Ordering::SeqCst);
        self.ctl.steps.store(0, Ordering::SeqCst);
        self.ctl.lr_scale_milli.store(1000, Ordering::SeqCst);
        {
            let mut s = self.shared.lock().unwrap();
            s.status = Status::Loading("Starte …".into());
            s.cfg = cfg.clone();
            s.epoch = 0;
            s.step = 0;
            s.hist = History::default();
            s.last_eval = None;
            s.elapsed = Duration::ZERO;
            s.samples_per_sec = 0.0;
            s.gflops = 0.0;
        }
        let (shared, ctl, data) = (self.shared.clone(), self.ctl.clone(), self.data.clone());
        self.handle = Some(std::thread::spawn(move || {
            let r = run(&cfg, &shared, &ctl, &data, &repaint);
            let mut s = shared.lock().unwrap();
            s.status = match r {
                Ok(true) => Status::Finished,
                Ok(false) => Status::Stopped,
                Err(e) => Status::Error(e),
            };
            drop(s);
            repaint();
        }));
    }
}

/// Lädt den Datensatz (oder nimmt den bereits geladenen).
pub fn ensure_data(spec: &DataSpec, data: &Mutex<Option<Arc<Dataset>>>, progress: &dyn Fn(String)) -> Result<Arc<Dataset>, String> {
    if let Some(d) = data.lock().unwrap().as_ref() {
        if d.spec == *spec {
            return Ok(d.clone());
        }
    }
    let d = Arc::new(data::load(spec, progress)?);
    *data.lock().unwrap() = Some(d.clone());
    Ok(d)
}

fn run(cfg: &TrainConfig, shared: &Mutex<Shared>, ctl: &Control, data: &Mutex<Option<Arc<Dataset>>>, repaint: &(dyn Fn() + Sync)) -> Result<bool, String> {
    let set_status = |st: Status| {
        shared.lock().unwrap().status = st;
        repaint();
    };
    let ds = ensure_data(&cfg.dataset, data, &|m| set_status(Status::Loading(m)))?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(cfg.threads.max(1)).build().map_err(|e| e.to_string())?;
    pool.install(|| train(cfg, &ds, shared, ctl, repaint))
}

fn train(cfg: &TrainConfig, ds: &Dataset, shared: &Mutex<Shared>, ctl: &Control, repaint: &(dyn Fn() + Sync)) -> Result<bool, String> {
    let k = ds.n_classes();
    let il = ds.img_len();
    let mut model = Model::build(&cfg.layers, cfg.input_shape(), k, cfg.init, cfg.seed)?;
    let mut rng = Rng::new(cfg.seed ^ 0xA5A5_5A5A);
    let n_train = cfg.train_size.clamp(cfg.batch.max(1), ds.n_train());
    // Feste Teilmenge des Trainingssets (reproduzierbar über den Seed)
    let mut pool_idx: Vec<usize> = (0..ds.n_train()).collect();
    rng.shuffle(&mut pool_idx);
    pool_idx.truncate(n_train);
    let bs = cfg.batch.max(1);
    let steps_per_epoch = n_train.div_ceil(bs);
    let macs = model.macs();
    // Feste Stichprobe für die schnelle Zwischen-Auswertung
    let quick: Vec<usize> = (0..ds.n_test()).step_by((ds.n_test() / 1000).max(1)).collect();
    {
        let mut s = shared.lock().unwrap();
        s.layer_names = model.layers.iter().map(|l| l.name()).collect();
        s.param_layers = model.layers.iter().enumerate().filter(|(_, l)| matches!(l, nn::Layer::Dense(_) | nn::Layer::Conv(_))).map(|(i, _)| i).collect();
        let nl = model.layers.len();
        s.hist.grad_norm = vec![Vec::new(); nl];
        s.hist.upd_ratio = vec![Vec::new(); nl];
        s.hist.act_std = vec![Vec::new(); nl];
        s.hist.act_dead = vec![Vec::new(); nl];
        s.steps_per_epoch = steps_per_epoch;
        s.macs = macs;
        s.n_params = model.n_params();
        s.model = Some(Arc::new(model.snapshot()));
        s.model_version += 1;
        s.status = if ctl.pause.load(Ordering::SeqCst) { Status::Paused } else { Status::Running };
    }
    repaint();

    let mut x = vec![0.0f32; bs * il];
    let mut y = vec![0u8; bs];
    let mut t: u64 = 0;
    let mut active = Duration::ZERO;
    let mut last_snap = Instant::now();
    let (mut win_t, mut win_n, mut win_busy) = (Instant::now(), 0usize, Duration::ZERO);
    let mut stats: Vec<ActStat> = Vec::new();

    for epoch in 0..cfg.epochs {
        rng.shuffle(&mut pool_idx);
        let (mut e_loss, mut e_correct, mut e_n) = (0.0f64, 0usize, 0usize);
        for (bi, chunk) in pool_idx.chunks(bs).enumerate() {
            // Pause / Einzelschritt / Stopp
            loop {
                if ctl.stop.load(Ordering::SeqCst) {
                    return Ok(false);
                }
                if !ctl.pause.load(Ordering::SeqCst) {
                    break;
                }
                if ctl.steps.load(Ordering::SeqCst) > 0 {
                    ctl.steps.fetch_sub(1, Ordering::SeqCst);
                    break;
                }
                {
                    let mut s = shared.lock().unwrap();
                    if s.status != Status::Paused {
                        s.status = Status::Paused;
                        drop(s);
                        repaint();
                    }
                }
                std::thread::sleep(Duration::from_millis(15));
            }
            let t0 = Instant::now();
            let b = chunk.len();
            for (j, &i) in chunk.iter().enumerate() {
                data::prepare(ds.train_image(i), &mut x[j * il..][..il], ds, cfg.norm, cfg.aug, Some(&mut rng));
                y[j] = ds.train_y[i];
            }
            let collect = t % 10 == 0;
            let logits = model.forward(&x[..b * il], b, true, if collect { Some(&mut stats) } else { None });
            let (loss, correct, grad) = nn::loss_grad(&logits, &y[..b], k, cfg.loss, cfg.smoothing);
            if !loss.is_finite() {
                return Err(format!("Der Verlust ist in Schritt {t} explodiert (NaN/∞). Lernrate zu hoch? Gradient Clipping probieren."));
            }
            model.backward(&grad, b);
            t += 1;
            let ep_f = epoch as f32 + bi as f32 / steps_per_epoch as f32;
            let lr = cfg.lr_at(ep_f) * ctl.lr_scale_milli.load(Ordering::Relaxed) as f32 / 1000.0;
            let st = nn::optimize(&mut model, &cfg.opt, lr, t, cfg.clip);
            active += t0.elapsed();
            win_busy += t0.elapsed();
            win_n += b;
            e_loss += loss as f64 * b as f64;
            e_correct += correct;
            e_n += b;

            let quick_eval = cfg.eval_every > 0 && t % cfg.eval_every as u64 == 0;
            let qe = quick_eval.then(|| evaluate(&mut model, ds, &quick, cfg, false));
            {
                let mut s = shared.lock().unwrap();
                let tx = t as f64;
                s.step = t;
                s.epoch = epoch;
                s.elapsed = active;
                let h = &mut s.hist;
                h.loss.push([tx, loss as f64]);
                h.acc.push([tx, correct as f64 / b as f64]);
                h.lr.push([tx, lr as f64]);
                h.grad_global.push([tx, st.global_grad_norm as f64]);
                for &(li, g) in &st.grad_norm {
                    h.grad_norm[li].push([tx, g as f64]);
                }
                for &(li, r) in &st.update_ratio {
                    h.upd_ratio[li].push([tx, r as f64]);
                }
                if collect {
                    for (li, a) in stats.iter().enumerate() {
                        h.act_std[li].push([tx, a.std as f64]);
                        if matches!(model.layers[li], nn::Layer::Act(_)) {
                            h.act_dead[li].push([tx, a.dead as f64]);
                        }
                    }
                }
                if let Some(e) = &qe {
                    h.val_loss.push([tx, e.loss as f64]);
                    h.val_acc.push([tx, e.acc as f64]);
                }
                if win_t.elapsed() > Duration::from_millis(500) && win_busy > Duration::ZERO {
                    s.samples_per_sec = win_n as f64 / win_busy.as_secs_f64();
                    s.gflops = s.samples_per_sec * macs as f64 * 2.0 * 3.0 / 1e9;
                    win_t = Instant::now();
                    win_n = 0;
                    win_busy = Duration::ZERO;
                }
                if last_snap.elapsed() > Duration::from_millis(200) || ctl.pause.load(Ordering::Relaxed) {
                    s.model = Some(Arc::new(model.snapshot()));
                    s.model_version += 1;
                    last_snap = Instant::now();
                }
                if s.status != Status::Running && !ctl.pause.load(Ordering::Relaxed) {
                    s.status = Status::Running;
                }
            }
            repaint();
            let delay = ctl.delay_ms.load(Ordering::Relaxed);
            if delay > 0 {
                std::thread::sleep(Duration::from_millis(delay as u64));
            }
        }
        // Epochenende: vollständige Auswertung auf dem Testset
        shared.lock().unwrap().status = Status::Loading(format!("Werte Epoche {} aus …", epoch + 1));
        repaint();
        let all: Vec<usize> = (0..ds.n_test()).collect();
        let ev = evaluate(&mut model, ds, &all, cfg, true);
        {
            let mut s = shared.lock().unwrap();
            let ex = (epoch + 1) as f64;
            let tx = t as f64;
            let h = &mut s.hist;
            h.epoch_train_loss.push([ex, e_loss / e_n.max(1) as f64]);
            h.epoch_train_acc.push([ex, e_correct as f64 / e_n.max(1) as f64]);
            h.epoch_val_loss.push([ex, ev.loss as f64]);
            h.epoch_val_acc.push([ex, ev.acc as f64]);
            h.val_loss.push([tx, ev.loss as f64]);
            h.val_acc.push([tx, ev.acc as f64]);
            s.last_eval = Some(ev);
            s.epoch = epoch + 1;
            s.model = Some(Arc::new(model.snapshot()));
            s.model_version += 1;
            s.status = if ctl.pause.load(Ordering::SeqCst) { Status::Paused } else { Status::Running };
        }
        repaint();
        win_t = Instant::now();
        win_n = 0;
        win_busy = Duration::ZERO;
    }
    Ok(true)
}

/// Wertet das Modell auf den angegebenen Testbildern aus (ohne Dropout, BN im Inferenzmodus).
pub fn evaluate(model: &mut Model, ds: &Dataset, idx: &[usize], cfg: &TrainConfig, full: bool) -> Eval {
    let bs = 500;
    let (k, il) = (ds.n_classes(), ds.img_len());
    let mut x = vec![0.0f32; bs * il];
    let mut conf = vec![vec![0u32; k]; k];
    let mut wrong = Vec::new();
    let (mut loss, mut correct) = (0.0f64, 0usize);
    let mut p = vec![0.0f32; k];
    for chunk in idx.chunks(bs) {
        let b = chunk.len();
        let mut y = Vec::with_capacity(b);
        for (j, &i) in chunk.iter().enumerate() {
            data::prepare(ds.test_image(i), &mut x[j * il..][..il], ds, cfg.norm, Augment::default(), None);
            y.push(ds.test_y[i]);
        }
        let z = model.forward(&x[..b * il], b, false, None);
        let (l, c, _) = nn::loss_grad(&z, &y, k, cfg.loss, 0.0);
        loss += l as f64 * b as f64;
        correct += c;
        if full {
            for (j, &i) in chunk.iter().enumerate() {
                nn::softmax(&z[j * k..][..k], &mut p);
                let pred = (0..k).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
                conf[y[j] as usize][pred] += 1;
                if pred != y[j] as usize {
                    wrong.push((i, pred as u8, y[j], p[pred]));
                }
            }
        }
    }
    // Die "überzeugtesten" Fehler zuerst
    wrong.sort_by(|a, b| b.3.total_cmp(&a.3));
    wrong.truncate(300);
    let n = idx.len().max(1) as f64;
    Eval { loss: (loss / n) as f32, acc: (correct as f64 / n) as f32, confusion: if full { conf } else { Vec::new() }, wrong }
}
