//! Berater: prüft die Konfiguration mit Praxisregeln, wertet einen
//! Trainingsverlauf aus und empfiehlt eine Einstellung für den Datensatz.

use crate::presets;
use crate::tutorial::Topic;
use nnlab::data::{Augment, DataKind};
use nnlab::nn::{ActKind, Init, LayerSpec, LossKind, Model, OptKind};
use nnlab::trainer::{History, Schedule, Status, TrainConfig};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Error,
    Warn,
    Tip,
    Good,
}

pub struct Fix {
    pub label: String,
    pub apply: Box<dyn Fn(&mut TrainConfig)>,
}

pub struct Advice {
    pub level: Level,
    pub title: String,
    pub text: String,
    pub fix: Option<Fix>,
    pub topic: Topic,
}

fn adv(level: Level, topic: Topic, title: impl Into<String>, text: impl Into<String>) -> Advice {
    Advice { level, title: title.into(), text: text.into(), fix: None, topic }
}

impl Advice {
    fn fix(mut self, label: impl Into<String>, f: impl Fn(&mut TrainConfig) + 'static) -> Self {
        self.fix = Some(Fix { label: label.into(), apply: Box::new(f) });
        self
    }
}

/// Was über den Datensatz bekannt ist (auch bevor er geladen ist).
pub struct DataInfo {
    pub kind: DataKind,
    pub shape: (usize, usize, usize),
    pub n_train: Option<usize>,
    pub classes: Option<usize>,
}

fn is_linear(l: &LayerSpec) -> bool {
    matches!(l, LayerSpec::Dense { .. } | LayerSpec::Conv { .. })
}

/// Empfohlene Einstellung für den Datensatz: (Rezeptname, Begründung, erwartete Genauigkeit).
pub fn ideal(info: &DataInfo) -> (&'static str, String) {
    let small = info.n_train.is_some_and(|n| n < 5000);
    match info.kind {
        _ if small => ("Gegen Overfitting: wenig Daten, viel Regularisierung", "Der Datensatz ist klein. Wichtiger als Kapazität ist hier Regularisierung: Dropout, Weight Decay, Augmentierung und Label Smoothing.".into()),
        DataKind::Digits => (
            "CNN modern: BatchNorm + AdamW + Cosine",
            "Ziffern sind einfach: Schon „Schnell & solide“ (MLP) erreicht ca. 98 %. Das moderne CNN mit leichter Verschiebungs-Augmentierung kommt auf ca. 99,3 % bis 99,5 %. Spiegeln wäre hier falsch, eine gespiegelte 2 ist keine 2.".into(),
        ),
        DataKind::Fashion => (
            "CNN modern: BatchNorm + AdamW + Cosine",
            "Fashion-MNIST braucht Formmerkmale: Ein MLP endet bei ca. 89 %, das moderne CNN bei ca. 92 % bis 93 %. Die meisten Restfehler sind Hemd ↔ T-Shirt/Pullover, die auch Menschen verwechseln.".into(),
        ),
        DataKind::Glyphs => (
            "CNN modern: BatchNorm + AdamW + Cosine",
            "Handschriftliche Zeichen (KMNIST, EMNIST) sind schwerer als Ziffern: MLP ca. 85 % bis 90 %, das moderne CNN ca. 93 % bis 96 %. Verschiebung ja, Spiegeln nein.".into(),
        ),
        DataKind::Photos => (
            "CNN modern: BatchNorm + AdamW + Cosine",
            "Fotos brauchen Faltungsnetze, Farbe und Augmentierung mit Spiegeln. Ein MLP schafft bei CIFAR-10 nur ca. 45 % bis 55 %, das breite CNN hier ca. 70 % bis 80 % nach 30 Epochen. Das kostet Rechenzeit: alle Threads verwenden.".into(),
        ),
        DataKind::Other => ("CNN modern: BatchNorm + AdamW + Cosine", "Für unbekannte Bilddaten ist ein kleines CNN mit BatchNorm ein robuster Start. Danach mit Augmentierung und Regularisierung nachjustieren.".into()),
    }
}

/// Prüft die Konfiguration, ohne zu trainieren.
pub fn check_config(c: &TrainConfig, info: &DataInfo, build: &Result<std::sync::Arc<Model>, String>, cores: usize) -> Vec<Advice> {
    let mut v = Vec::new();
    let l = &c.layers;
    let adaptive = matches!(c.opt.kind, OptKind::Adam | OptKind::AdamW | OptKind::RmsProp);
    let acts: Vec<ActKind> = l.iter().filter_map(|x| if let LayerSpec::Act(a) = x { Some(*a) } else { None }).collect();
    let n_linear = l.iter().filter(|x| is_linear(x)).count();
    let has_conv = l.iter().any(|x| matches!(x, LayerSpec::Conv { .. }));
    let has_bn = l.iter().any(|x| matches!(x, LayerSpec::BatchNorm));
    let has_dropout = l.iter().any(|x| matches!(x, LayerSpec::Dropout { .. }));
    let n_train = info.n_train.map(|n| c.train_size.min(n)).unwrap_or(c.train_size);

    if let Err(e) = build {
        if !e.is_empty() {
            v.push(adv(Level::Error, Topic::Architecture, "Netz lässt sich nicht bauen", e.clone()).fix("Architektur „MLP 128“", |c| c.layers = presets::arch_presets()[1].layers.clone()));
        }
    }

    // --- Architektur ---
    // Zwei lineare Schichten ohne Aktivierung dazwischen
    let mut last_linear: Option<usize> = None;
    for (i, x) in l.iter().enumerate() {
        match x {
            LayerSpec::Act(_) => last_linear = None,
            _ if is_linear(x) => {
                if let Some(j) = last_linear {
                    let ins = i;
                    v.push(
                        adv(Level::Warn, Topic::Activation, format!("Keine Aktivierung zwischen Schicht {} und {}", j + 1, i + 1), "Zwei lineare Schichten ohne Nichtlinearität dazwischen sind mathematisch eine einzige lineare Schicht: W₂(W₁x) = (W₂W₁)x. Die zusätzliche Schicht bringt nichts außer Rechenzeit.")
                            .fix(format!("ReLU vor Schicht {} einfügen", i + 1), move |c| c.layers.insert(ins, LayerSpec::Act(ActKind::ReLU))),
                    );
                }
                last_linear = Some(i);
            }
            _ => {}
        }
    }
    if let Some(j) = last_linear {
        if n_linear > 0 && matches!(l[j], LayerSpec::Dense { .. } | LayerSpec::Conv { .. }) && !l[j + 1..].iter().any(|x| matches!(x, LayerSpec::Act(_))) {
            let ins = j + 1;
            v.push(
                adv(Level::Warn, Topic::Activation, format!("Schicht {} endet ohne Aktivierung", j + 1), "Direkt danach folgt die (lineare) Ausgabeschicht, beide fallen zu einer zusammen.")
                    .fix("ReLU anhängen", move |c| c.layers.insert(ins, LayerSpec::Act(ActKind::ReLU))),
            );
        }
    }
    for w in l.windows(2).enumerate() {
        if let (i, [LayerSpec::Act(_), LayerSpec::Act(_)]) = w {
            v.push(adv(Level::Tip, Topic::Activation, format!("Zwei Aktivierungen hintereinander (Schicht {} und {})", i + 1, i + 2), "Das ist selten beabsichtigt. Eine davon kann weg.").fix(format!("Schicht {} entfernen", i + 2), move |c| {
                c.layers.remove(i + 1);
            }));
        }
    }
    if n_linear == 0 {
        v.push(adv(Level::Tip, Topic::Architecture, "Kein verdecktes Netz", "Ohne verdeckte Schicht ist das Modell linear (multinomiale logistische Regression). Auf MNIST liegt die Obergrenze bei ca. 92 %. Ein guter Vergleichswert, aber kein ernsthaftes Netz.").fix("MLP 128 verwenden", |c| c.layers = presets::arch_presets()[1].layers.clone()));
    }
    let deep_sat = acts.iter().filter(|a| matches!(a, ActKind::Sigmoid | ActKind::Tanh)).count();
    if deep_sat >= 3 && !has_bn {
        v.push(
            adv(Level::Warn, Topic::Activation, format!("{deep_sat} sättigende Aktivierungen ohne BatchNorm"), "Sigmoid und Tanh haben kleine Ableitungen (Sigmoid höchstens 0,25). Über viele Schichten multipliziert verschwindet der Gradient in den vorderen Schichten (Vanishing Gradients).")
                .fix("Alle auf ReLU umstellen (+ He-Init)", |c| {
                    for x in c.layers.iter_mut() {
                        if let LayerSpec::Act(a) = x {
                            if matches!(a, ActKind::Sigmoid | ActKind::Tanh) {
                                *a = ActKind::ReLU;
                            }
                        }
                    }
                    c.init = Init::He;
                }),
        );
    }
    if let (Some(k), Some((i, u))) = (info.classes, l.iter().enumerate().rev().find_map(|(i, x)| if let LayerSpec::Dense { units } = x { Some((i, *units)) } else { None })) {
        if u < k {
            v.push(adv(Level::Warn, Topic::Dense, format!("Engpass: Schicht {} hat nur {u} Neuronen für {k} Klassen", i + 1), "Die letzte verdeckte Schicht ist schmaler als die Zahl der Klassen. Alle Information muss durch diesen Flaschenhals.").fix(format!("Auf {} Neuronen verbreitern", (k * 4).max(64)), move |c| c.layers[i] = LayerSpec::Dense { units: (k * 4).max(64) }));
        }
    }
    if let Some((i, p)) = l.iter().enumerate().find_map(|(i, x)| if let LayerSpec::Dropout { p } = x { (*p > 0.6).then_some((i, *p)) } else { None }) {
        v.push(adv(Level::Warn, Topic::Dropout, format!("Sehr hoher Dropout ({p:.2}) in Schicht {}", i + 1), "Mehr als 60 % der Neuronen fallen bei jedem Schritt aus. Das Netz lernt dann nur schleppend (Underfitting).").fix("Auf 0,3 senken", move |c| c.layers[i] = LayerSpec::Dropout { p: 0.3 }));
    }
    if has_bn && c.batch < 16 {
        v.push(adv(Level::Warn, Topic::BatchNorm, format!("BatchNorm mit Batchgröße {}", c.batch), "BatchNorm schätzt Mittelwert und Varianz aus dem Batch. Bei so kleinen Batches sind diese Schätzungen sehr verrauscht, bei Batch 1 ist die Varianz sogar 0.").fix("Batch 64", |c| c.batch = 64));
    }
    if matches!(info.kind, DataKind::Photos) && !has_conv {
        v.push(adv(Level::Warn, Topic::Conv, "Fotos ohne Faltungsschichten", "Ein MLP kennt keine Nachbarschaft zwischen Pixeln und muss jedes Muster an jeder Stelle neu lernen. Bei Fotos wie CIFAR-10 bleibt es bei ca. 50 %.").fix("Breites CNN verwenden", |c| c.layers = presets::modern_cnn(true)));
    } else if matches!(info.kind, DataKind::Fashion | DataKind::Glyphs) && !has_conv && n_linear > 0 {
        v.push(adv(Level::Tip, Topic::Conv, "Ein CNN wäre hier deutlich besser", "Bei Kleidung und handgeschriebenen Zeichen bringen Faltungen mehrere Prozentpunkte gegenüber einem MLP.").fix("Modernes CNN verwenden", |c| c.layers = presets::modern_cnn(false)));
    }
    if let Ok(m) = build {
        let p = m.n_params();
        if p > 30 * n_train && !has_dropout && c.opt.weight_decay == 0.0 && c.aug.shift == 0 {
            v.push(
                adv(Level::Warn, Topic::TrainSize, format!("{} Parameter für {} Trainingsbilder", crate::fmt_n(p), n_train), "Das Netz hat ein Vielfaches mehr Parameter als Beispiele und keine Regularisierung. Es kann die Trainingsbilder auswendig lernen (Overfitting).")
                    .fix("Augmentierung ±2 px + Weight Decay", |c| {
                        c.aug.shift = 2;
                        if c.opt.kind == OptKind::Adam {
                            c.opt.kind = OptKind::AdamW;
                        }
                        c.opt.weight_decay = if c.opt.kind == OptKind::AdamW { 0.02 } else { 5e-4 };
                    }),
            );
        }
        let flops_epoch = m.macs() as f64 * 6.0 * n_train as f64;
        let secs = flops_epoch * c.epochs as f64 / (c.threads.max(1) as f64 * 8e9);
        if secs > 600.0 {
            v.push(adv(Level::Tip, Topic::Throughput, format!("Geschätzte Dauer: ca. {:.0} Minuten", secs / 60.0), "Grobe Schätzung bei etwa 8 GFLOP/s pro Kern. Für erste Versuche weniger Epochen oder Trainingsbilder nehmen."));
        }
    }

    // --- Initialisierung ---
    if c.init == Init::Zero {
        v.push(adv(Level::Error, Topic::Init, "Null-Initialisierung", "Alle Neuronen einer Schicht starten gleich, bekommen denselben Gradienten und bleiben für immer identisch. Mit ReLU fließt sogar gar kein Gradient in die verdeckten Schichten.").fix("He-Init", |c| c.init = Init::He));
    } else if c.init == Init::Small && n_linear >= 3 {
        v.push(adv(Level::Warn, Topic::Init, "Sehr kleine Startgewichte in tiefem Netz", "Mit σ = 0,01 schrumpfen die Aktivierungen von Schicht zu Schicht, das Signal kommt hinten kaum an.").fix("He-Init", |c| c.init = Init::He));
    } else if c.init == Init::He && deep_sat > 0 && !acts.iter().any(|a| matches!(a, ActKind::ReLU | ActKind::LeakyReLU | ActKind::GELU)) {
        v.push(adv(Level::Tip, Topic::Init, "He-Init mit Sigmoid/Tanh", "He ist für ReLU berechnet (Faktor 2 wegen der abgeschnittenen Hälfte). Für Sigmoid und Tanh passt Xavier besser.").fix("Xavier-Init", |c| c.init = Init::Xavier));
    } else if c.init == Init::Xavier && acts.iter().all(|a| matches!(a, ActKind::ReLU)) && !acts.is_empty() && n_linear >= 4 {
        v.push(adv(Level::Tip, Topic::Init, "Xavier-Init mit ReLU", "In tiefen ReLU-Netzen schrumpft das Signal mit Xavier pro Schicht etwas. He ist dafür gemacht.").fix("He-Init", |c| c.init = Init::He));
    }

    // --- Verlust ---
    if c.loss == LossKind::Mse {
        v.push(adv(Level::Tip, Topic::Loss, "MSE statt Kreuzentropie", "Für Klassifikation lernt Kreuzentropie schneller: Ihr Gradient ist bei sicheren Fehlern groß, der von MSE fast null.").fix("Kreuzentropie", |c| c.loss = LossKind::CrossEntropy));
    }
    if c.smoothing > 0.2 {
        v.push(adv(Level::Warn, Topic::LabelSmoothing, format!("Starkes Label Smoothing (ε = {:.2})", c.smoothing), "Die Zielverteilung wird so flach, dass das Netz kaum noch eindeutige Entscheidungen lernt.").fix("ε = 0,1", |c| c.smoothing = 0.1));
    }

    // --- Optimierer und Lernrate ---
    let lr = c.lr;
    if adaptive && lr > 0.01 {
        v.push(adv(Level::Warn, Topic::LearningRate, format!("η = {lr:.1e} ist für {} sehr hoch", c.opt.kind.name()), "Adaptive Verfahren machen pro Parameter Schritte der Größe ≈ η. Über 0,01 wird das Training meist instabil, ReLUs sterben.").fix("η = 0,001", |c| c.lr = 0.001));
    } else if adaptive && lr < 1e-5 {
        v.push(adv(Level::Warn, Topic::LearningRate, format!("η = {lr:.1e} ist sehr klein"), "Das Netz wird kaum vorankommen.").fix("η = 0,001", |c| c.lr = 0.001));
    } else if !adaptive && lr > 0.5 {
        v.push(adv(Level::Warn, Topic::LearningRate, format!("η = {lr:.2} ist für SGD sehr hoch"), "Das Training wird vermutlich divergieren (Verlust springt oder wird NaN).").fix("η = 0,05", |c| c.lr = 0.05));
    } else if !adaptive && lr < 1e-3 {
        v.push(adv(Level::Warn, Topic::LearningRate, format!("η = {lr:.1e} ist für SGD sehr klein"), "SGD macht genau η·Gradient große Schritte. So klein dauert es sehr lange.").fix("η = 0,05", |c| c.lr = 0.05));
    }
    if c.opt.kind == OptKind::Sgd {
        v.push(adv(Level::Tip, Topic::Momentum, "SGD ohne Momentum", "Momentum 0,9 beschleunigt fast immer und glättet das Zickzack in schmalen Tälern.").fix("Momentum 0,9 (η halbieren)", |c| {
            c.opt.kind = OptKind::Momentum;
            c.opt.momentum = 0.9;
            c.lr = (c.lr * 0.5).max(1e-3);
        }));
    }
    if matches!(c.opt.kind, OptKind::Momentum | OptKind::Nesterov) && c.opt.momentum > 0.97 {
        v.push(adv(Level::Warn, Topic::Momentum, "Momentum sehr nah an 1", "Die effektive Lernrate wird η/(1−μ), also sehr groß, und die Kurve schwingt.").fix("μ = 0,9", |c| c.opt.momentum = 0.9));
    }
    if c.opt.kind == OptKind::Adam && c.opt.weight_decay > 1e-3 {
        v.push(adv(Level::Tip, Topic::WeightDecay, "Weight Decay mit Adam", "Bei Adam wird L2-Regularisierung durch √v geteilt und wirkt deshalb ungleichmäßig. AdamW entkoppelt das.").fix("AdamW verwenden", |c| c.opt.kind = OptKind::AdamW));
    }
    if c.opt.weight_decay > (if c.opt.kind == OptKind::AdamW { 0.2 } else { 0.01 }) {
        v.push(adv(Level::Warn, Topic::WeightDecay, "Sehr starker Weight Decay", "Die Gewichte werden so stark zur Null gezogen, dass das Netz kaum noch etwas lernen kann.").fix("Moderat einstellen", |c| c.opt.weight_decay = if c.opt.kind == OptKind::AdamW { 0.02 } else { 5e-4 }));
    }
    if let Some(cl) = c.clip {
        if cl < 0.1 {
            v.push(adv(Level::Tip, Topic::GradClip, "Sehr enges Gradient Clipping", "Fast jeder Schritt wird gekürzt, das wirkt wie eine viel kleinere Lernrate.").fix("Grenze 1,0", |c| c.clip = Some(1.0)));
        }
    }

    // --- Plan ---
    if c.schedule == Schedule::Step && c.step_every >= c.epochs {
        v.push(adv(Level::Tip, Topic::Schedule, "Stufenplan ohne Stufe", format!("Die erste Stufe käme erst nach Epoche {}, das Training endet vorher.", c.step_every)).fix("Stufe nach 1/3 der Epochen", |c| c.step_every = (c.epochs / 3).max(1)));
    }
    if c.warmup_epochs >= c.epochs as f32 * 0.5 {
        v.push(adv(Level::Warn, Topic::Warmup, "Warmup dauert über die Hälfte des Trainings", "Die Lernrate erreicht ihren Zielwert kaum.").fix("Warmup 1 Epoche", |c| c.warmup_epochs = 1.0_f32.min(c.epochs as f32 * 0.2)));
    }
    if c.schedule == Schedule::Constant && c.epochs >= 15 {
        v.push(adv(Level::Tip, Topic::Schedule, "Lange Läufe profitieren von einem Plan", "Gegen Ende kleinere Schritte bringen oft noch einige Zehntelprozent.").fix("Cosine-Plan", |c| {
            c.schedule = Schedule::Cosine;
            c.lr_min_frac = 0.02;
        }));
    }

    // --- Daten und Rechnen ---
    if c.aug.flip && matches!(info.kind, DataKind::Digits | DataKind::Glyphs) {
        v.push(adv(Level::Warn, Topic::Augmentation, "Spiegeln bei Ziffern oder Buchstaben", "Gespiegelt wird aus einer 2 keine 2 und aus einem b ein d. Die Augmentierung erzeugt falsch gelabelte Beispiele.").fix("Spiegeln aus", |c| c.aug.flip = false));
    }
    if c.aug.shift > 3 && info.shape.1 <= 32 {
        v.push(adv(Level::Tip, Topic::Augmentation, format!("Verschiebung ±{} px bei {} px Bildgröße", c.aug.shift, info.shape.1), "Teile der Figur fallen aus dem Bild. 1 bis 2 Pixel reichen meist.").fix("±2 px", |c| c.aug.shift = 2));
    }
    if n_train < 5000 && !has_dropout && c.opt.weight_decay == 0.0 && c.aug == Augment::default() && n_linear > 0 {
        v.push(adv(Level::Tip, Topic::TrainSize, format!("Nur {n_train} Trainingsbilder, keine Regularisierung"), "Mit so wenig Daten ist Overfitting wahrscheinlich. Augmentierung ist hier die wirksamste Gegenmaßnahme.").fix("Augmentierung ±2 px", |c| c.aug.shift = 2));
    }
    if c.norm != nnlab::data::Norm::Standard {
        v.push(adv(Level::Tip, Topic::Normalization, "Eingabe nicht standardisiert", "Mit Mittelwert 0 und Streuung 1 pro Kanal lernen Netze meist schneller und stabiler.").fix("Standardisieren", |c| c.norm = nnlab::data::Norm::Standard));
    }
    if c.batch <= 4 {
        v.push(adv(Level::Tip, Topic::BatchSize, format!("Batchgröße {}", c.batch), "Sehr verrauschte Schritte und kaum Parallelität: Der Rechner ist schlecht ausgelastet.").fix("Batch 64", |c| c.batch = 64));
    } else if c.batch >= 512 && !adaptive {
        v.push(adv(Level::Tip, Topic::BatchSize, format!("Große Batches ({}) mit SGD", c.batch), "Pro Epoche gibt es nur wenige Schritte. Mit SGD die Lernrate etwa proportional zur Batchgröße anheben und Warmup verwenden."));
    }
    if c.threads < cores {
        v.push(adv(Level::Tip, Topic::Threads, format!("{} von {cores} CPU-Threads", c.threads), "Mehr Threads beschleunigen das Training.").fix("Alle Threads", move |c| c.threads = cores));
    }
    if c.eval_every > 0 && c.eval_every < 20 {
        v.push(adv(Level::Tip, Topic::EvalEvery, "Sehr häufige Zwischenauswertung", "Jede Auswertung kostet Rechenzeit, die dem Training fehlt.").fix("Alle 100 Schritte", |c| c.eval_every = 100));
    }

    if !v.iter().any(|a| a.level <= Level::Warn) {
        v.push(adv(Level::Good, Topic::Advisor, "Keine Probleme in der Konfiguration gefunden", "Die Einstellungen sind in sich stimmig. Ob sie gut sind, zeigt das Training: Danach wertet der Berater auch den Verlauf aus."));
    }
    v.sort_by_key(|a| a.level);
    v
}

fn median_last(v: &[[f64; 2]], n: usize) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut t: Vec<f64> = v[v.len().saturating_sub(n)..].iter().map(|p| p[1]).filter(|x| x.is_finite()).collect();
    if t.is_empty() {
        return None;
    }
    t.sort_by(|a, b| a.total_cmp(b));
    Some(t[t.len() / 2])
}

/// Wertet einen Trainingsverlauf aus. `cfg` ist die Konfiguration des Laufs.
pub fn check_training(cfg: &TrainConfig, h: &History, status: &Status, names: &[String], param_layers: &[usize], classes: usize, kind: DataKind) -> Vec<Advice> {
    let mut v = Vec::new();
    if let Status::Error(e) = status {
        if e.contains("NaN") {
            v.push(adv(Level::Error, Topic::LearningRate, "Training explodiert", "Der Verlust wurde unendlich bzw. NaN. Fast immer ist die Lernrate zu hoch, seltener fehlt Gradient Clipping.").fix("η durch 10 teilen + Clipping 1,0", |c| {
                c.lr /= 10.0;
                c.clip = Some(1.0);
            }));
        }
    }
    if h.loss.len() < 30 {
        return v;
    }
    let chance = (classes as f64).ln();
    let loss_now = median_last(&h.loss, 50).unwrap_or(f64::NAN);
    let steps = h.loss.len();

    // Kein Lernfortschritt
    if steps > 200 && loss_now > chance * 0.9 {
        v.push(adv(Level::Error, Topic::LearningRate, "Das Netz lernt praktisch nichts", format!("Der Verlust liegt nach {steps} Schritten noch bei {loss_now:.2}, Raten ergäbe ln({classes}) = {chance:.2}. Ursachen: Lernrate viel zu klein oder zu groß, Null-Initialisierung, verschwindende Gradienten.")));
    }
    // Instabilität: starke Sprünge nach oben
    let tail: Vec<f64> = h.loss[h.loss.len().saturating_sub(200)..].iter().map(|p| p[1]).collect();
    if tail.len() > 50 {
        let first = median_last(&h.loss[..h.loss.len() / 2], 50).unwrap_or(loss_now);
        if loss_now > first * 1.3 && loss_now > 0.2 {
            v.push(adv(Level::Warn, Topic::LearningRate, "Der Trainingsverlust steigt wieder", format!("Median zuletzt {loss_now:.3} gegenüber {first:.3} vorher. Das deutet auf eine zu hohe Lernrate.")).fix("η halbieren", |c| c.lr *= 0.5));
        }
    }

    // Overfitting / Underfitting über die Epochenwerte
    if let (Some(tr), Some(te)) = (h.epoch_train_acc.last(), h.epoch_val_acc.last()) {
        let gap = (tr[1] - te[1]) * 100.0;
        let best_vl = h.epoch_val_loss.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
        let last_vl = h.epoch_val_loss.last().map(|p| p[1]).unwrap_or(best_vl);
        let best_ep = h.epoch_val_loss.iter().min_by(|a, b| a[1].total_cmp(&b[1])).map(|p| p[0] as usize).unwrap_or(0);
        if gap > 3.0 || (h.epoch_val_loss.len() >= 4 && last_vl > best_vl * 1.15) {
            v.push(
                adv(
                    Level::Warn,
                    Topic::TrainSize,
                    "Overfitting",
                    format!("Train {:.1} % gegenüber Test {:.1} % (Lücke {gap:.1} Prozentpunkte). Der Test-Verlust war in Epoche {best_ep} am niedrigsten ({best_vl:.3}) und liegt jetzt bei {last_vl:.3}. Das Netz lernt Trainingsbilder auswendig.", tr[1] * 100.0, te[1] * 100.0),
                )
                .fix("Regularisierung: Dropout 0,3, Augmentierung, Weight Decay", |c| {
                    if !c.layers.iter().any(|l| matches!(l, LayerSpec::Dropout { .. })) {
                        if let Some(i) = c.layers.iter().rposition(|l| matches!(l, LayerSpec::Act(_))) {
                            c.layers.insert(i + 1, LayerSpec::Dropout { p: 0.3 });
                        }
                    }
                    c.aug.shift = c.aug.shift.max(2);
                    if c.opt.kind == OptKind::Adam {
                        c.opt.kind = OptKind::AdamW;
                    }
                    c.opt.weight_decay = c.opt.weight_decay.max(if c.opt.kind == OptKind::AdamW { 0.02 } else { 5e-4 });
                }),
            );
        }
        let target = match kind {
            DataKind::Digits => 97.5,
            DataKind::Fashion => 87.0,
            DataKind::Glyphs => 85.0,
            DataKind::Photos => 55.0,
            DataKind::Other => 0.0,
        };
        if tr[1] * 100.0 < target && gap < 2.0 && h.epoch_train_acc.len() >= 3 {
            v.push(adv(Level::Tip, Topic::Architecture, "Underfitting", format!("Schon die Train-Genauigkeit ({:.1} %) ist niedrig, und Train und Test liegen dicht beieinander. Das Netz ist zu klein, lernt zu langsam oder ist zu stark regularisiert. Mehr Kapazität (breiter, tiefer, CNN) oder mehr Epochen helfen.", tr[1] * 100.0)));
        }
        if gap.abs() < 1.5 && te[1] * 100.0 >= target && target > 0.0 {
            v.push(adv(Level::Good, Topic::ViewAcc, format!("Gutes Ergebnis: {:.2} % auf dem Testset", te[1] * 100.0), "Train und Test liegen dicht beieinander, das Netz generalisiert."));
        }
    }

    // Update-Verhältnis
    let mut ratios = Vec::new();
    for &li in param_layers {
        if let Some(r) = h.upd_ratio.get(li).and_then(|v| median_last(v, 100)) {
            ratios.push((li, r));
        }
    }
    if let Some(&(li, r)) = ratios.iter().max_by(|a, b| a.1.total_cmp(&b.1)) {
        if r > 2e-2 {
            v.push(adv(Level::Warn, Topic::ViewUpdateRatio, format!("Große Updates in Schicht {} ({})", li + 1, names.get(li).cloned().unwrap_or_default()), format!("‖ΔW‖/‖W‖ ≈ {r:.1e} pro Schritt (Faustregel: 1e-3). Die Lernrate ist für diese Schicht zu hoch.")).fix("η durch 3 teilen", |c| c.lr /= 3.0));
        }
    }
    if let Some(&(li, r)) = ratios.iter().min_by(|a, b| a.1.total_cmp(&b.1)) {
        if r < 1e-5 && r > 0.0 {
            v.push(adv(Level::Warn, Topic::ViewUpdateRatio, format!("Schicht {} lernt kaum ({})", li + 1, names.get(li).cloned().unwrap_or_default()), format!("‖ΔW‖/‖W‖ ≈ {r:.1e} pro Schritt. Lernrate zu klein oder Gradienten verschwinden.")));
        }
    }
    // Vanishing Gradients: erste vs. letzte Gewichtsschicht
    if param_layers.len() >= 3 {
        let g = |li: usize| h.grad_norm.get(li).and_then(|v| median_last(v, 100));
        if let (Some(a), Some(b)) = (g(param_layers[0]), g(*param_layers.last().unwrap())) {
            if a > 0.0 && b / a > 100.0 {
                v.push(adv(Level::Warn, Topic::ViewGradNorm, "Verschwindende Gradienten", format!("Der Gradient der ersten Gewichtsschicht ist {:.0}× kleiner als der der Ausgabeschicht. Die vorderen Schichten lernen kaum. Abhilfe: ReLU statt Sigmoid/Tanh, BatchNorm, passende Initialisierung.", b / a)).fix("BatchNorm nach jeder Dense/Conv einfügen", |c| {
                    let mut out = Vec::new();
                    for (i, l) in c.layers.iter().enumerate() {
                        out.push(*l);
                        let lin = matches!(l, LayerSpec::Dense { .. } | LayerSpec::Conv { .. });
                        if lin && !matches!(c.layers.get(i + 1), Some(LayerSpec::BatchNorm)) {
                            out.push(LayerSpec::BatchNorm);
                        }
                    }
                    c.layers = out;
                }));
            }
        }
    }
    // Tote ReLUs / Sättigung
    for (li, d) in h.act_dead.iter().enumerate() {
        let Some(m) = median_last(d, 20) else { continue };
        let name = names.get(li).cloned().unwrap_or_default();
        if name == "ReLU" && m > 0.85 {
            v.push(adv(Level::Warn, Topic::ViewActStats, format!("Tote ReLUs in Schicht {}", li + 1), format!("{:.0} % der Einheiten sind auf diesem Batch 0. Wahrscheinlich sind viele Neuronen dauerhaft „gestorben“.", m * 100.0)).fix("LeakyReLU verwenden", |c| {
                for l in c.layers.iter_mut() {
                    if let LayerSpec::Act(a) = l {
                        if *a == ActKind::ReLU {
                            *a = ActKind::LeakyReLU;
                        }
                    }
                }
            }));
        }
        if (name == "Sigmoid" || name == "Tanh") && m > 0.5 {
            v.push(adv(Level::Warn, Topic::ViewActStats, format!("Gesättigte Einheiten in Schicht {}", li + 1), format!("{:.0} % der Einheiten sind gesättigt, ihre Ableitung ist fast 0.", m * 100.0)));
        }
    }
    let _ = cfg;
    v.sort_by_key(|a| a.level);
    v
}
