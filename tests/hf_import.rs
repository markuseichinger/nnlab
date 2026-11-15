//! Prüft den Hugging-Face-Import offline mit nachgebauten Dateien im
//! HF-Format (Parquet mit Bild-Struktur {bytes, path}, Snappy/Zstd, PNG/JPEG).

use nnlab::data::{self, DataSpec, HfLayout};
use std::path::Path;

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn json(name: &str) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(fixture(name)).unwrap()).unwrap()
}

#[test]
fn info_findet_bild_und_label() {
    let l = data::parse_info(&json("info_cifar.json")).unwrap();
    assert_eq!(l.config, "plain_text");
    assert_eq!(l.image_col, "img");
    assert_eq!(l.label_col, "label");
    assert_eq!(l.classes.len(), 10);
    assert_eq!(l.classes[9], "truck");
}

#[test]
fn text_datensatz_wird_abgelehnt() {
    let e = data::parse_info(&json("info_text.json")).unwrap_err();
    assert!(e.contains("Image"), "{e}");
}

#[test]
fn parquet_liste() {
    let (tr, te) = data::parquet_urls(&json("parquet_mnist.json"), "mnist").unwrap();
    assert!(tr[0].ends_with("train/0.parquet"));
    assert!(te[0].ends_with("test/0.parquet"));
    assert!(data::parquet_urls(&json("parquet_mnist.json"), "gibtsnicht").is_err());
}

fn layout() -> HfLayout {
    HfLayout { config: "default".into(), image_col: "img".into(), label_col: "label".into(), classes: vec!["a".into(), "b".into(), "c".into()] }
}

#[test]
fn rgb_parquet_mit_snappy_und_zstd() {
    // 40×40 RGB wird auf 32×32 verkleinert
    let spec = DataSpec::hf("test/rgb", 32, true);
    let ds = data::dataset_from_parquet(&spec, &layout(), &fixture("rgb_train.parquet"), Some(&fixture("rgb_test.parquet"))).unwrap();
    assert_eq!((ds.c, ds.h, ds.w), (3, 32, 32));
    assert_eq!(ds.n_train(), 60);
    assert_eq!(ds.n_test(), 30);
    assert_eq!(&ds.train_y[..6], &[0, 1, 2, 0, 1, 2]);
    // Klasse 1: weißer Streifen im mittleren Drittel, Hintergrund (100,155,50)
    let img = ds.train_image(1);
    let n = 32 * 32;
    let px = |ch: usize, x: usize, y: usize| img[ch * n + y * 32 + x];
    assert!(px(0, 16, 10) > 240 && px(1, 16, 10) > 240 && px(2, 16, 10) > 240, "Streifen weiß");
    assert!((px(0, 2, 10) as i32 - 100).abs() < 6 && (px(1, 2, 10) as i32 - 155).abs() < 6 && (px(2, 2, 10) as i32 - 50).abs() < 6, "Hintergrund");
    assert_eq!(ds.mean.len(), 3);
}

#[test]
fn graustufen_ohne_testsplit() {
    let spec = DataSpec::hf("test/gray", 28, false);
    let ds = data::dataset_from_parquet(&spec, &layout(), &fixture("gray_train.parquet"), None).unwrap();
    assert_eq!((ds.c, ds.h, ds.w), (1, 28, 28));
    // 15 % werden als Testset abgezweigt
    assert_eq!(ds.n_test(), 45 * 15 / 100);
    assert_eq!(ds.n_train() + ds.n_test(), 45);
}

#[test]
fn rgb_als_graustufen() {
    let spec = DataSpec::hf("test/rgb", 16, false);
    let ds = data::dataset_from_parquet(&spec, &layout(), &fixture("rgb_train.parquet"), None).unwrap();
    assert_eq!((ds.c, ds.h, ds.w), (1, 16, 16));
}

#[test]
fn cnn_trainiert_auf_importierten_farbbildern() {
    use nnlab::nn::{ActKind, LayerSpec};
    use nnlab::trainer::{Status, TrainConfig, Trainer};
    let spec = DataSpec::hf("test/rgb", 16, true);
    let ds = data::dataset_from_parquet(&spec, &layout(), &fixture("rgb_train.parquet"), Some(&fixture("rgb_test.parquet"))).unwrap();
    let mut cfg = TrainConfig { dataset: spec, train_size: 60, epochs: 6, batch: 8, threads: 2, eval_every: 0, ..TrainConfig::default() };
    cfg.layers = vec![LayerSpec::Conv { filters: 4, k: 3, same: true }, LayerSpec::BatchNorm, LayerSpec::Act(ActKind::ReLU), LayerSpec::MaxPool { size: 2 }, LayerSpec::Dense { units: 16 }, LayerSpec::Act(ActKind::ReLU)];
    cfg.lr = 0.01;
    let mut tr = Trainer::new();
    *tr.data.lock().unwrap() = Some(std::sync::Arc::new(ds));
    tr.start(cfg, false, || {});
    loop {
        std::thread::sleep(std::time::Duration::from_millis(50));
        let s = tr.shared.lock().unwrap();
        match &s.status {
            Status::Finished => {
                let acc = s.last_eval.as_ref().unwrap().acc;
                assert_eq!(s.model.as_ref().unwrap().classes, 3);
                assert!(acc > 0.95, "Testgenauigkeit nur {acc}");
                break;
            }
            Status::Error(e) => panic!("{e}"),
            _ => {}
        }
    }
}
