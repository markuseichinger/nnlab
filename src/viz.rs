//! Farbskalen, Kachelbilder und Kurvenaufbereitung für die Anzeige.

use eframe::egui::{Color32, ColorImage};

/// Hintergrund der Kachelraster.
pub const GAP: Color32 = Color32::from_rgb(12, 13, 17);

/// Divergierende Skala für Werte in [−1, 1]: blau – fast schwarz – orange.
pub fn diverging(v: f32) -> Color32 {
    let v = v.clamp(-1.0, 1.0);
    let (r0, g0, b0) = (22.0, 24.0, 30.0);
    let (r1, g1, b1) = if v >= 0.0 { (255.0, 150.0, 50.0) } else { (70.0, 160.0, 255.0) };
    let t = v.abs().powf(0.75);
    Color32::from_rgb((r0 + (r1 - r0) * t) as u8, (g0 + (g1 - g0) * t) as u8, (b0 + (b1 - b0) * t) as u8)
}

/// Sequenzielle Skala (Viridis-ähnlich) für Werte in [0, 1].
pub fn sequential(v: f32) -> Color32 {
    const STOPS: [(f32, [f32; 3]); 6] = [
        (0.0, [16.0, 14.0, 30.0]),
        (0.2, [59.0, 28.0, 140.0]),
        (0.4, [33.0, 102.0, 172.0]),
        (0.6, [31.0, 158.0, 137.0]),
        (0.8, [120.0, 209.0, 81.0]),
        (1.0, [253.0, 231.0, 37.0]),
    ];
    let v = v.clamp(0.0, 1.0);
    for w in STOPS.windows(2) {
        let ((a, ca), (b, cb)) = (w[0], w[1]);
        if v <= b {
            let t = (v - a) / (b - a);
            let c = |i: usize| (ca[i] + (cb[i] - ca[i]) * t) as u8;
            return Color32::from_rgb(c(0), c(1), c(2));
        }
    }
    Color32::from_rgb(253, 231, 37)
}

pub fn gray(v: f32) -> Color32 {
    let g = (v.clamp(0.0, 1.0) * 255.0) as u8;
    Color32::from_rgb(g, g, g)
}

/// Wie Werte eingefärbt werden.
#[derive(Clone, Copy)]
pub enum Scale {
    /// Symmetrisch um 0, normiert auf max |v|
    Signed(f32),
    /// 0 … max
    Positive(f32),
}

impl Scale {
    /// Wählt die Skala passend zu den Daten.
    pub fn auto(vals: &[f32]) -> Scale {
        let mn = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let amax = mn.abs().max(mx.abs()).max(1e-8);
        if mn < -1e-6 * amax { Scale::Signed(amax) } else { Scale::Positive(mx.max(1e-8)) }
    }
    pub fn color(self, v: f32) -> Color32 {
        match self {
            Scale::Signed(m) => diverging(v / m),
            Scale::Positive(m) => sequential(v / m),
        }
    }
}

/// Ordnet `n` Kacheln der Größe tw×th in `cols` Spalten mit 1 px Abstand an.
/// `value(kachel, x, y)` liefert den Wert, `scale` die Farbe.
pub fn tiles(n: usize, tw: usize, th: usize, cols: usize, scale: impl Fn(usize) -> Scale, value: impl Fn(usize, usize, usize) -> f32) -> ColorImage {
    let cols = cols.clamp(1, n.max(1));
    let rows = n.div_ceil(cols).max(1);
    let (w, h) = (cols * (tw + 1) + 1, rows * (th + 1) + 1);
    let mut px = vec![GAP; w * h];
    for t in 0..n {
        let (ox, oy) = (1 + (t % cols) * (tw + 1), 1 + (t / cols) * (th + 1));
        let sc = scale(t);
        for y in 0..th {
            for x in 0..tw {
                px[(oy + y) * w + ox + x] = sc.color(value(t, x, y));
            }
        }
    }
    ColorImage::new([w, h], px)
}

/// Ein Vektor als Raster aus Einzelpixeln (cols breit).
pub fn vector(vals: &[f32], cols: usize, scale: Scale) -> ColorImage {
    let cols = cols.clamp(1, vals.len().max(1));
    let rows = vals.len().div_ceil(cols).max(1);
    let mut px = vec![GAP; cols * rows];
    for (i, &v) in vals.iter().enumerate() {
        px[i] = scale.color(v);
    }
    ColorImage::new([cols, rows], px)
}

/// Graustufenbild 28×28 aus u8.
pub fn digit(img: &[u8], side: usize) -> ColorImage {
    ColorImage::new([side, side], img.iter().map(|&p| gray(p as f32 / 255.0)).collect())
}

/// Exponentiell geglättete Kurve, auf höchstens `max_pts` Punkte ausgedünnt.
/// `alpha` = Glättung (0 = keine). Optional log10 der y-Werte.
pub fn curve(data: &[[f64; 2]], alpha: f64, max_pts: usize, log: bool) -> Vec<[f64; 2]> {
    if data.is_empty() {
        return Vec::new();
    }
    let step = data.len().div_ceil(max_pts.max(1)).max(1);
    let mut out = Vec::with_capacity(data.len() / step + 2);
    let mut ema = data[0][1];
    for (i, p) in data.iter().enumerate() {
        ema = if alpha > 0.0 { alpha * ema + (1.0 - alpha) * p[1] } else { p[1] };
        if i % step == 0 || i + 1 == data.len() {
            let y = if log { ema.max(1e-12).log10() } else { ema };
            if y.is_finite() {
                out.push([p[0], y]);
            }
        }
    }
    out
}

/// Achsenbeschriftung für log10-transformierte Werte.
pub fn log_label(v: f64) -> String {
    let e = v.round();
    if (v - e).abs() < 1e-6 { format!("1e{}", e as i32) } else { String::new() }
}

/// Unterscheidbare Farben für Schichten.
pub fn series_color(i: usize) -> Color32 {
    const C: [Color32; 10] = [
        Color32::from_rgb(255, 150, 50),
        Color32::from_rgb(70, 160, 255),
        Color32::from_rgb(120, 209, 81),
        Color32::from_rgb(230, 90, 160),
        Color32::from_rgb(253, 220, 60),
        Color32::from_rgb(160, 120, 255),
        Color32::from_rgb(60, 210, 200),
        Color32::from_rgb(240, 90, 80),
        Color32::from_rgb(180, 180, 180),
        Color32::from_rgb(200, 160, 110),
    ];
    C[i % C.len()]
}
