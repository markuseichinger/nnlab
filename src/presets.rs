//! Vordefinierte Einstellungen. Ein Rezept setzt die komplette Konfiguration
//! neu (nur Datensatz und CPU-Threads bleiben), räumt also auch ein
//! „verkonfiguriertes“ Netz vollständig auf.

use nnlab::data::{Augment, DataKind};
use nnlab::nn::{ActKind, Init, LayerSpec, LossKind, OptKind};
use nnlab::trainer::{Schedule, TrainConfig};

pub struct Recipe {
    pub name: &'static str,
    pub group: Group,
    pub info: &'static str,
    pub apply: fn(&mut TrainConfig),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Group {
    Recommended,
    Lesson,
}

/// Grundzustand: alles auf Standard, Datensatz und Threads bleiben.
pub fn base(cfg: &mut TrainConfig) {
    *cfg = TrainConfig { dataset: cfg.dataset.clone(), threads: cfg.threads, ..TrainConfig::default() };
}

fn photos(cfg: &TrainConfig) -> bool {
    cfg.dataset.kind() == DataKind::Photos
}

use ActKind::*;
use LayerSpec::*;

fn mlp(widths: &[usize], act: ActKind, dropout: f32) -> Vec<LayerSpec> {
    let mut v = Vec::new();
    for &w in widths {
        v.push(Dense { units: w });
        v.push(Act(act));
        if dropout > 0.0 {
            v.push(Dropout { p: dropout });
        }
    }
    v
}

/// Kleiner, moderner CNN; für Fotos etwas breiter.
pub fn modern_cnn(wide: bool) -> Vec<LayerSpec> {
    let (a, b, c) = if wide { (32, 64, 128) } else { (16, 32, 64) };
    vec![
        Conv { filters: a, k: 3, same: true },
        BatchNorm,
        Act(ReLU),
        MaxPool { size: 2 },
        Conv { filters: b, k: 3, same: true },
        BatchNorm,
        Act(ReLU),
        MaxPool { size: 2 },
        Conv { filters: c, k: 3, same: true },
        BatchNorm,
        Act(ReLU),
        MaxPool { size: 2 },
        Dense { units: 128 },
        BatchNorm,
        Act(ReLU),
        Dropout { p: 0.3 },
    ]
}

pub fn recipes() -> Vec<Recipe> {
    vec![
        Recipe {
            name: "Standard (alles zurücksetzen)",
            group: Group::Recommended,
            info: "Der Ausgangszustand: MLP mit 128 ReLU-Neuronen, Adam, η = 0,001, 10 Epochen. Setzt jede Verstellung zurück.",
            apply: base,
        },
        Recipe {
            name: "Schnell & solide: MLP + Adam",
            group: Group::Recommended,
            info: "256 ReLU-Neuronen mit etwas Dropout, Adam. Trainiert in Sekunden und erreicht auf MNIST ca. 98 %.",
            apply: |c| {
                base(c);
                c.layers = mlp(&[256], ReLU, 0.2);
                c.batch = 128;
                c.epochs = 10;
            },
        },
        Recipe {
            name: "Klassisch: SGD + Momentum + Stufenplan",
            group: Group::Recommended,
            info: "So wurde jahrelang trainiert: SGD mit Momentum 0,9, L2-Regularisierung und Lernratenstufen. Etwas langsamer als Adam, generalisiert aber oft besser.",
            apply: |c| {
                base(c);
                c.layers = mlp(&[256, 256], ReLU, 0.0);
                c.opt.kind = OptKind::Momentum;
                c.opt.momentum = 0.9;
                c.lr = 0.05;
                c.opt.weight_decay = 5e-4;
                c.schedule = Schedule::Step;
                c.step_every = 4;
                c.step_factor = 0.3;
                c.epochs = 12;
                c.batch = 64;
            },
        },
        Recipe {
            name: "CNN (LeNet-artig)",
            group: Group::Recommended,
            info: "Zwei Faltungsblöcke mit 5×5-Kernen und Pooling. Der Klassiker von 1998, auf MNIST ca. 99 %.",
            apply: |c| {
                base(c);
                c.layers = vec![Conv { filters: 8, k: 5, same: true }, Act(ReLU), MaxPool { size: 2 }, Conv { filters: 16, k: 5, same: true }, Act(ReLU), MaxPool { size: 2 }, Dense { units: 64 }, Act(ReLU)];
                c.epochs = 8;
            },
        },
        Recipe {
            name: "CNN modern: BatchNorm + AdamW + Cosine",
            group: Group::Recommended,
            info: "Drei Faltungsblöcke mit BatchNorm, Dropout, AdamW mit entkoppeltem Weight Decay, Warmup, Cosine-Plan, Label Smoothing und Augmentierung. Das beste Rezept hier, aber rechenintensiv.",
            apply: |c| {
                let ph = photos(c);
                base(c);
                c.layers = modern_cnn(ph);
                c.opt.kind = OptKind::AdamW;
                c.lr = 0.002;
                c.opt.weight_decay = 0.02;
                c.schedule = Schedule::Cosine;
                c.lr_min_frac = 0.02;
                c.warmup_epochs = 1.0;
                c.smoothing = 0.05;
                c.aug = Augment { shift: 2, flip: ph };
                c.batch = 128;
                c.epochs = if ph { 30 } else { 15 };
            },
        },
        Recipe {
            name: "Gegen Overfitting: wenig Daten, viel Regularisierung",
            group: Group::Recommended,
            info: "Nur 3000 Trainingsbilder. Dropout, Weight Decay, Augmentierung und Label Smoothing halten die Lücke zwischen Train und Test klein.",
            apply: |c| {
                let ph = photos(c);
                base(c);
                c.train_size = 3000;
                c.layers = mlp(&[256], ReLU, 0.4);
                c.opt.kind = OptKind::AdamW;
                c.opt.weight_decay = 0.05;
                c.aug = Augment { shift: 2, flip: ph };
                c.smoothing = 0.1;
                c.epochs = 40;
                c.eval_every = 20;
            },
        },
        Recipe {
            name: "Maximale Rechenlast",
            group: Group::Recommended,
            info: "Breiter CNN, große Batches, alle Kerne. Gut, um in der Aktivitätsanzeige zu sehen, was der Rechner leistet.",
            apply: |c| {
                base(c);
                c.layers = modern_cnn(true);
                c.batch = 256;
                c.epochs = 5;
                c.threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
            },
        },
        Recipe {
            name: "Vanishing Gradients (tiefes Sigmoid-Netz)",
            group: Group::Lesson,
            info: "Sechs Sigmoid-Schichten mit SGD. Im Tab „Gradienten-Dynamik“ sind die Gradienten der vorderen Schichten um Größenordnungen kleiner, das Netz lernt kaum. Danach: Aktivierungen auf ReLU stellen oder BatchNorm einfügen.",
            apply: |c| {
                base(c);
                c.layers = mlp(&[64, 64, 64, 64, 64, 64], Sigmoid, 0.0);
                c.opt.kind = OptKind::Sgd;
                c.lr = 0.1;
                c.init = Init::Xavier;
                c.epochs = 5;
            },
        },
        Recipe {
            name: "Zu hohe Lernrate",
            group: Group::Lesson,
            info: "SGD mit η = 1,5: Der Verlust springt wild oder explodiert zu NaN. Danach η schrittweise verkleinern oder Gradient Clipping einschalten.",
            apply: |c| {
                base(c);
                c.opt.kind = OptKind::Sgd;
                c.lr = 1.5;
                c.epochs = 3;
            },
        },
        Recipe {
            name: "Zu kleine Lernrate",
            group: Group::Lesson,
            info: "SGD mit η = 0,0001: Alles ist stabil, aber der Verlust sinkt quälend langsam. Mit dem Live-Faktor oben lässt sich η während des Trainings anheben.",
            apply: |c| {
                base(c);
                c.opt.kind = OptKind::Sgd;
                c.lr = 1e-4;
                c.epochs = 5;
            },
        },
        Recipe {
            name: "Overfitting erzwingen",
            group: Group::Lesson,
            info: "Großes Netz, nur 1000 Bilder, keine Regularisierung, 60 Epochen. Die Train-Genauigkeit erreicht 100 %, der Test-Verlust steigt wieder an. Vergleiche mit „Gegen Overfitting“.",
            apply: |c| {
                base(c);
                c.train_size = 1000;
                c.layers = mlp(&[512, 512], ReLU, 0.0);
                c.epochs = 60;
                c.eval_every = 16;
            },
        },
        Recipe {
            name: "Symmetrie: Null-Initialisierung",
            group: Group::Lesson,
            info: "Alle Gewichte starten bei 0. Jedes Neuron einer Schicht bekommt denselben Gradienten und bleibt identisch, das Netz verhält sich wie ein einziges Neuron. Im Tab „Gewichte“ sind alle Bilder gleich.",
            apply: |c| {
                base(c);
                c.layers = mlp(&[64], Tanh, 0.0);
                c.init = Init::Zero;
                c.opt.kind = OptKind::Sgd;
                c.lr = 0.1;
                c.epochs = 3;
            },
        },
        Recipe {
            name: "Tote ReLUs",
            group: Group::Lesson,
            info: "Adam mit viel zu hoher Lernrate schiebt viele ReLU-Neuronen in den negativen Bereich, wo sie nie wieder aktiv werden. Im Tab „Gradienten-Dynamik“ steigt der Anteil inaktiver Einheiten. Gegenmittel: LeakyReLU oder kleineres η.",
            apply: |c| {
                base(c);
                c.layers = mlp(&[256, 256], ReLU, 0.0);
                c.lr = 0.03;
                c.epochs = 3;
            },
        },
        Recipe {
            name: "Batchgröße 1 (reines SGD)",
            group: Group::Lesson,
            info: "Ein Bild pro Schritt: sehr verrauschte Gradienten, kaum Parallelität, der Durchsatz bricht ein. Vergleiche Bilder/s und CPU-Last mit Batch 256.",
            apply: |c| {
                base(c);
                c.batch = 1;
                c.opt.kind = OptKind::Sgd;
                c.lr = 0.01;
                c.train_size = 10_000;
                c.epochs = 2;
                c.eval_every = 1000;
            },
        },
        Recipe {
            name: "MSE statt Kreuzentropie",
            group: Group::Lesson,
            info: "Mittlerer quadratischer Fehler nach Softmax: Bei falschen, sehr sicheren Vorhersagen wird der Gradient winzig. Das Netz lernt deutlich langsamer als mit Kreuzentropie.",
            apply: |c| {
                base(c);
                c.loss = LossKind::Mse;
                c.opt.kind = OptKind::Sgd;
                c.lr = 0.5;
                c.epochs = 5;
            },
        },
    ]
}

/// Nur die Architektur (für den Architektur-Editor).
pub struct ArchPreset {
    pub name: &'static str,
    pub info: &'static str,
    pub layers: Vec<LayerSpec>,
}

pub fn arch_presets() -> Vec<ArchPreset> {
    vec![
        ArchPreset { name: "Softmax-Regression (keine verdeckte Schicht)", info: "Ein lineares Modell. Auf MNIST ca. 92 %.", layers: vec![] },
        ArchPreset { name: "MLP 128", info: "Eine verdeckte Schicht.", layers: mlp(&[128], ReLU, 0.0) },
        ArchPreset { name: "MLP 2×256 + Dropout", info: "Zwei verdeckte Schichten mit Dropout.", layers: mlp(&[256, 256], ReLU, 0.2) },
        ArchPreset { name: "CNN LeNet-artig", info: "Zwei Faltungsblöcke.", layers: vec![Conv { filters: 8, k: 5, same: true }, Act(ReLU), MaxPool { size: 2 }, Conv { filters: 16, k: 5, same: true }, Act(ReLU), MaxPool { size: 2 }, Dense { units: 64 }, Act(ReLU)] },
        ArchPreset { name: "CNN modern (BN + Dropout)", info: "Drei Faltungsblöcke mit BatchNorm.", layers: modern_cnn(false) },
        ArchPreset { name: "CNN breit (für Fotos)", info: "Wie oben, mit 32/64/128 Filtern.", layers: modern_cnn(true) },
    ]
}
