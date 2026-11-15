//! Training ohne Oberfläche, z. B. zum Messen der Rechenleistung:
//!   cargo run --release --example bench -- [mlp|cnn|sgd] [Epochen] [Threads] [Trainingsbilder] [fashion]
use nnlab::nn::{ActKind, LayerSpec, OptKind};
use nnlab::trainer::{Status, TrainConfig, Trainer};
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arch = args.get(1).map(String::as_str).unwrap_or("mlp");
    let mut cfg = TrainConfig::default();
    cfg.epochs = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    if let Some(t) = args.get(3).and_then(|s| s.parse().ok()) {
        cfg.threads = t;
    }
    if let Some(n) = args.get(4).and_then(|s| s.parse().ok()) {
        cfg.train_size = n;
    }
    if args.iter().any(|a| a == "fashion") {
        cfg.dataset = nnlab::data::DataSpec::fashion();
    }
    cfg.layers = match arch {
        "cnn" => vec![
            LayerSpec::Conv { filters: 16, k: 3, same: true },
            LayerSpec::BatchNorm,
            LayerSpec::Act(ActKind::ReLU),
            LayerSpec::MaxPool { size: 2 },
            LayerSpec::Conv { filters: 32, k: 3, same: true },
            LayerSpec::BatchNorm,
            LayerSpec::Act(ActKind::ReLU),
            LayerSpec::MaxPool { size: 2 },
            LayerSpec::Dense { units: 128 },
            LayerSpec::Act(ActKind::ReLU),
            LayerSpec::Dropout { p: 0.3 },
        ],
        "sgd" => {
            cfg.opt.kind = OptKind::Momentum;
            cfg.lr = 0.05;
            vec![LayerSpec::Dense { units: 128 }, LayerSpec::Act(ActKind::ReLU)]
        }
        _ => vec![LayerSpec::Dense { units: 128 }, LayerSpec::Act(ActKind::ReLU)],
    };
    println!("Architektur {arch}, {} Epochen, {} Threads", cfg.epochs, cfg.threads);
    let mut tr = Trainer::new();
    let t0 = Instant::now();
    tr.start(cfg, false, || {});
    let mut last_epoch = usize::MAX;
    loop {
        std::thread::sleep(Duration::from_millis(500));
        let s = tr.shared.lock().unwrap();
        if s.epoch != last_epoch && !s.hist.epoch_val_acc.is_empty() && s.hist.epoch_val_acc.len() == s.epoch {
            last_epoch = s.epoch;
            let h = &s.hist;
            let (tl, ta, vl, va) = (h.epoch_train_loss.last().unwrap()[1], h.epoch_train_acc.last().unwrap()[1], h.epoch_val_loss.last().unwrap()[1], h.epoch_val_acc.last().unwrap()[1]);
            println!(
                "Epoche {:>2}: Train-Verlust {tl:.4}  Train-Acc {:.2}%  Test-Verlust {vl:.4}  Test-Acc {:.2}%   {:.0} Samples/s  {:.1} GFLOP/s  ({:.1}s)",
                s.epoch, ta * 100.0, va * 100.0, s.samples_per_sec, s.gflops, t0.elapsed().as_secs_f64()
            );
        }
        match &s.status {
            Status::Finished | Status::Stopped => break,
            Status::Error(e) => {
                println!("Fehler: {e}");
                break;
            }
            _ => {}
        }
    }
}
