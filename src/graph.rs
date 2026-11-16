//! Netzgraph: Neuronen als Knoten, Gewichte als Kanten, live aus dem Modell.
//!
//! Jede Spalte ist ein „Block“: eine lineare Schicht (Dense oder Conv) samt
//! der folgenden Aktivierung, BatchNorm, Pooling und Dropout. Knoten sind bei
//! flachen Schichten einzelne Neuronen, bei Faltungen ganze Kanäle
//! (Feature-Maps). Kanten fassen bei räumlichen Daten alle Gewichte zwischen
//! zwei Kanälen zusammen.

use nnlab::nn::{Layer, Model, Shape};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EdgeMode {
    /// Das Gewicht selbst
    Weight,
    /// Beitrag zum Eingang des Zielneurons für das aktuelle Bild: w · a
    Signal,
    /// Letzter Gradient ∂L/∂w
    Gradient,
    /// Änderung seit dem vorigen Schnappschuss (wo gerade gelernt wird)
    Change,
}

impl EdgeMode {
    pub const ALL: [EdgeMode; 4] = [EdgeMode::Weight, EdgeMode::Signal, EdgeMode::Gradient, EdgeMode::Change];
    pub fn name(self) -> &'static str {
        match self {
            EdgeMode::Weight => "Gewichte",
            EdgeMode::Signal => "Signalfluss (w · a) für dieses Bild",
            EdgeMode::Gradient => "Gradienten ∂L/∂w",
            EdgeMode::Change => "Änderung live (ΔW)",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pick {
    /// Gleichmäßig verteilt (stabil während des Trainings)
    Spread,
    /// Die bei diesem Bild am stärksten aktiven
    Active,
}

pub struct Column {
    pub title: String,
    pub detail: String,
    /// Index der linearen Schicht, die diese Spalte erzeugt (None = Eingabe)
    pub layer: Option<usize>,
    /// Ausgabeform des Blocks
    pub shape: Shape,
    /// Anzahl Knoten insgesamt (Neuronen bzw. Kanäle)
    pub n_total: usize,
    /// Welche davon gezeigt werden
    pub idx: Vec<usize>,
    /// Ein Wert pro gezeigtem Knoten (Aktivierung bzw. Mittel der Feature-Map)
    pub val: Vec<f32>,
    /// Feature-Maps der gezeigten Knoten (bei räumlichen Spalten)
    pub maps: Vec<Vec<f32>>,
    /// Vollständige Ausgabe des Blocks (für den Signalfluss)
    pub out: Vec<f32>,
    pub is_output: bool,
}

impl Column {
    pub fn spatial(&self) -> bool {
        self.shape.spatial()
    }
}

fn is_linear(l: &Layer) -> bool {
    matches!(l, Layer::Dense(_) | Layer::Conv(_))
}

/// Wählt bis zu `max` Knoten aus `n` aus.
fn choose(n: usize, max: usize, pick: Pick, score: impl Fn(usize) -> f32) -> Vec<usize> {
    if n <= max {
        return (0..n).collect();
    }
    match pick {
        Pick::Spread => (0..max).map(|i| i * n / max).collect(),
        Pick::Active => {
            let mut v: Vec<usize> = (0..n).collect();
            v.sort_by(|&a, &b| score(b).abs().total_cmp(&score(a).abs()));
            v.truncate(max);
            v.sort();
            v
        }
    }
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len().max(1) as f32
}

/// Teilt das Netz in Spalten auf.
pub fn columns(model: &Model, input: &[f32], outs: &[Vec<f32>], probs: &[f32], max_nodes: usize, pick: Pick) -> Vec<Column> {
    let mut cols = Vec::new();
    // Eingabe: ein Knoten pro Kanal
    let s = model.input;
    let hw = s.h * s.w;
    cols.push(Column {
        title: "Eingabe".into(),
        detail: s.describe(),
        layer: None,
        shape: s,
        n_total: s.c,
        idx: (0..s.c).collect(),
        val: (0..s.c).map(|c| mean(&input[c * hw..][..hw])).collect(),
        maps: (0..s.c).map(|c| input[c * hw..][..hw].to_vec()).collect(),
        out: input.to_vec(),
        is_output: false,
    });
    let n = model.layers.len();
    let mut i = 0;
    while i < n {
        if !is_linear(&model.layers[i]) {
            // Nichtlineare Schichten vor der ersten linearen: ignorieren
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i;
        while end + 1 < n && !is_linear(&model.layers[end + 1]) {
            end += 1;
        }
        let shape = model.layers[end].out_shape();
        let out = &outs[end];
        let is_output = end + 1 == n;
        let parts: Vec<String> = model.layers[start..=end].iter().map(|l| l.name()).collect();
        let (n_total, hw) = (shape.c, shape.h * shape.w);
        let val_all: Vec<f32> = if is_output { probs.to_vec() } else { (0..n_total).map(|c| mean(&out[c * hw..][..hw])).collect() };
        let idx = if is_output { (0..n_total).collect() } else { choose(n_total, max_nodes, pick, |k| val_all[k]) };
        cols.push(Column {
            title: if is_output { "Ausgabe".into() } else { parts[0].clone() },
            detail: if parts.len() > 1 { format!("{} → {}", parts[1..].join(" → "), shape.describe()) } else { shape.describe() },
            layer: Some(start),
            shape,
            n_total,
            val: idx.iter().map(|&k| val_all[k]).collect(),
            maps: if shape.spatial() { idx.iter().map(|&k| out[k * hw..][..hw].to_vec()).collect() } else { Vec::new() },
            idx,
            out: out.clone(),
            is_output,
        });
        i = end + 1;
    }
    cols
}

/// Kante: (Quellknoten, Zielknoten, Wert mit Vorzeichen).
pub type Edge = (usize, usize, f32);

/// Gewichte eines Ausgabeneurons/-kanals `o` zu einem Eingangskanal `c`.
fn slice<'a>(l: &Layer, v: &'a [f32], o: usize, c: usize, src_hw: usize) -> &'a [f32] {
    match l {
        Layer::Dense(d) => &v[o * d.inp + c * src_hw..][..src_hw],
        Layer::Conv(cv) => {
            let kk = cv.k * cv.k;
            &v[o * cv.cin * kk + c * kk..][..kk]
        }
        _ => &[],
    }
}

/// Fasst einen Gewichtsblock zu einem Wert zusammen: einzelnes Gewicht direkt,
/// sonst L2-Norm mit dem Vorzeichen der Summe.
fn agg(w: &[f32]) -> f32 {
    if w.len() == 1 {
        return w[0];
    }
    let n = w.iter().map(|x| x * x).sum::<f32>().sqrt();
    if w.iter().sum::<f32>() < 0.0 { -n } else { n }
}

/// Kanten zwischen Spalte `b-1` und `b`, je Zielknoten die `top_k` stärksten
/// (bzw. alle Kanten des ausgewählten Knotens).
pub fn edges(model: &Model, prev: Option<&Model>, cols: &[Column], b: usize, mode: EdgeMode, top_k: usize, focus: Option<(usize, usize)>) -> Vec<Edge> {
    let (src, dst) = (&cols[b - 1], &cols[b]);
    let Some(li) = dst.layer else { return Vec::new() };
    let l = &model.layers[li];
    let ps = l.params();
    let (w, g) = (&ps[0].w, &ps[0].g);
    let prev_w = prev.and_then(|p| p.layers.get(li)).map(|pl| pl.params()).filter(|pp| !pp.is_empty() && pp[0].w.len() == w.len()).map(|pp| pp[0].w.clone());
    if mode == EdgeMode::Change && prev_w.is_none() {
        return Vec::new();
    }
    let src_hw = src.shape.h * src.shape.w;
    let mut out = Vec::new();
    for (dj, &o) in dst.idx.iter().enumerate() {
        let mut row: Vec<Edge> = src
            .idx
            .iter()
            .enumerate()
            .map(|(si, &c)| {
                let ws = slice(l, w, o, c, src_hw);
                let v = match mode {
                    EdgeMode::Weight => agg(ws),
                    EdgeMode::Gradient => agg(slice(l, g, o, c, src_hw)),
                    EdgeMode::Change => {
                        let pw = slice(l, prev_w.as_ref().unwrap(), o, c, src_hw);
                        ws.iter().zip(pw).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt()
                    }
                    EdgeMode::Signal => match l {
                        // exakter Beitrag dieses Eingangskanals zum Eingang des Neurons
                        Layer::Dense(_) => ws.iter().zip(&src.out[c * src_hw..][..src_hw]).map(|(a, x)| a * x).sum(),
                        // Mittel über alle Positionen ≈ Summe der Kernelgewichte × Mittel der Eingangs-Map
                        _ => ws.iter().sum::<f32>() * mean(&src.out[c * src_hw..][..src_hw]),
                    },
                };
                (si, dj, v)
            })
            .collect();
        let focused = focus.is_some_and(|(fc, fn_)| (fc == b && fn_ == dj) || fc == b - 1);
        if !focused && row.len() > top_k {
            row.sort_by(|a, b| b.2.abs().total_cmp(&a.2.abs()));
            row.truncate(top_k);
        }
        if let Some((fc, fn_)) = focus {
            if fc == b - 1 {
                row.retain(|e| e.0 == fn_);
            }
        }
        out.extend(row);
    }
    out
}

/// Gewichte, die in ein Neuron/einen Kanal eingehen, als Bild (für die Detailansicht).
/// Liefert (Breite, Höhe, Kanäle, Werte planar) oder None.
pub fn receptive_field(model: &Model, li: usize, o: usize) -> Option<(usize, usize, usize, Vec<f32>)> {
    // Eingabeform der Schicht li
    let inp = if li == 0 { model.input } else { model.layers[li - 1].out_shape() };
    match &model.layers[li] {
        Layer::Dense(d) if inp.spatial() => Some((inp.w, inp.h, inp.c, d.w.w[o * d.inp..][..d.inp].to_vec())),
        Layer::Conv(c) => {
            let ckk = c.cin * c.k * c.k;
            Some((c.k, c.k, c.cin, c.w.w[o * ckk..][..ckk].to_vec()))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnlab::nn::{ActKind, Init, LayerSpec};

    fn model() -> Model {
        let specs = [
            LayerSpec::Conv { filters: 4, k: 3, same: true },
            LayerSpec::Act(ActKind::ReLU),
            LayerSpec::MaxPool { size: 2 },
            LayerSpec::Dense { units: 50 },
            LayerSpec::Act(ActKind::Tanh),
        ];
        Model::build(&specs, Shape::new(1, 8, 8), 3, Init::He, 1).unwrap()
    }

    #[test]
    fn spalten_und_kanten() {
        let mut m = model();
        let x: Vec<f32> = (0..64).map(|i| (i % 7) as f32 / 7.0).collect();
        let outs = m.forward_trace(&x);
        let probs = vec![0.2, 0.3, 0.5];
        let cols = columns(&m, &x, &outs, &probs, 16, Pick::Spread);
        assert_eq!(cols.len(), 4); // Eingabe, Conv-Block, Dense-Block, Ausgabe
        assert_eq!(cols[1].n_total, 4);
        assert!(cols[1].spatial());
        assert_eq!(cols[2].n_total, 50);
        assert_eq!(cols[2].idx.len(), 16);
        assert!(cols[3].is_output);
        // Conv: 1 Eingangskanal → 4 Filter, je Ziel höchstens top_k
        let e = edges(&m, None, &cols, 1, EdgeMode::Weight, 8, None);
        assert_eq!(e.len(), 4);
        // Dense nach Pooling: Quellknoten sind die 4 Kanäle (je 4×4 Positionen)
        let e = edges(&m, None, &cols, 2, EdgeMode::Weight, 2, None);
        assert_eq!(e.len(), 16 * 2);
        // Signalfluss Dense → Ausgabe ist exakt w · a
        let e = edges(&m, None, &cols, 3, EdgeMode::Signal, 100, None);
        let Layer::Dense(d) = &m.layers[5] else { panic!() };
        let (si, dj, v) = e[0];
        let src = cols[2].idx[si];
        assert!((v - d.w.w[dj * d.inp + src] * cols[2].out[src]).abs() < 1e-6);
        // Fokus auf einen Quellknoten: nur dessen ausgehende Kanten, alle Ziele
        let e = edges(&m, None, &cols, 3, EdgeMode::Weight, 1, Some((2, 0)));
        assert_eq!(e.len(), 3);
        assert!(e.iter().all(|e| e.0 == 0));
        // Ohne vorigen Schnappschuss gibt es keine Änderung
        assert!(edges(&m, None, &cols, 3, EdgeMode::Change, 5, None).is_empty());
        let mut m2 = m.clone();
        if let Layer::Dense(d) = &mut m2.layers[5] {
            d.w.w[0] += 0.5;
        }
        let e = edges(&m2, Some(&m), &cols, 3, EdgeMode::Change, 100, None);
        let mx = e.iter().map(|e| e.2).fold(0.0, f32::max);
        assert!((mx - 0.5).abs() < 1e-6);
        assert!(receptive_field(&m, 0, 1).is_some_and(|r| r.0 == 3 && r.2 == 1));
    }
}
