//! Zeichenfläche: eigene Ziffern malen und wie bei MNIST aufbereiten
//! (in 20×20 einpassen, im 28×28-Feld über den Schwerpunkt zentrieren).

pub const N: usize = 280;

pub struct Canvas {
    pub buf: Vec<f32>,
    pub last: Option<(f32, f32)>,
    pub version: u64,
    pub brush: f32,
}

impl Default for Canvas {
    fn default() -> Self {
        Self { buf: vec![0.0; N * N], last: None, version: 1, brush: 11.0 }
    }
}

impl Canvas {
    pub fn clear(&mut self) {
        self.buf.iter_mut().for_each(|v| *v = 0.0);
        self.version += 1;
    }

    pub fn is_empty(&self) -> bool {
        !self.buf.iter().any(|&v| v > 0.05)
    }

    /// Zieht einen weichen Strich von der letzten Position nach (x, y) (0…N).
    pub fn stroke_to(&mut self, x: f32, y: f32) {
        let (x0, y0) = self.last.unwrap_or((x, y));
        let dist = ((x - x0).powi(2) + (y - y0).powi(2)).sqrt();
        let steps = (dist / 2.0).ceil().max(1.0) as usize;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            self.dab(x0 + (x - x0) * t, y0 + (y - y0) * t);
        }
        self.last = Some((x, y));
        self.version += 1;
    }

    fn dab(&mut self, cx: f32, cy: f32) {
        let r = self.brush;
        let (xa, xb) = ((cx - r - 2.0).max(0.0) as usize, ((cx + r + 2.0) as usize).min(N - 1));
        let (ya, yb) = ((cy - r - 2.0).max(0.0) as usize, ((cy + r + 2.0) as usize).min(N - 1));
        for y in ya..=yb {
            for x in xa..=xb {
                let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                let v = (1.0 - (d - r * 0.6) / (r * 0.4)).clamp(0.0, 1.0);
                let p = &mut self.buf[y * N + x];
                *p = p.max(v);
            }
        }
    }

    /// MNIST-artige Aufbereitung auf 28×28 Graustufen (u8).
    pub fn to_mnist(&self) -> Vec<u8> {
        let mut out = vec![0u8; 28 * 28];
        let (mut x0, mut y0, mut x1, mut y1) = (N, N, 0, 0);
        for y in 0..N {
            for x in 0..N {
                if self.buf[y * N + x] > 0.05 {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }
        if x1 < x0 {
            return out;
        }
        let (w, h) = ((x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32);
        let scale = 20.0 / w.max(h);
        let (tw, th) = (((w * scale).round() as usize).clamp(1, 20), ((h * scale).round() as usize).clamp(1, 20));
        // Flächenmittelung in die Zielzellen
        let mut acc = vec![0.0f32; tw * th];
        let mut cnt = vec![0.0f32; tw * th];
        for y in y0..=y1 {
            for x in x0..=x1 {
                let tx = (((x - x0) as f32 + 0.5) * scale).floor().min(tw as f32 - 1.0) as usize;
                let ty = (((y - y0) as f32 + 0.5) * scale).floor().min(th as f32 - 1.0) as usize;
                acc[ty * tw + tx] += self.buf[y * N + x];
                cnt[ty * tw + tx] += 1.0;
            }
        }
        let small: Vec<f32> = acc.iter().zip(&cnt).map(|(a, c)| if *c > 0.0 { a / c } else { 0.0 }).collect();
        // Schwerpunkt auf (14, 14) legen
        let (mut sx, mut sy, mut sm) = (0.0, 0.0, 0.0);
        for y in 0..th {
            for x in 0..tw {
                let v = small[y * tw + x];
                sx += v * x as f32;
                sy += v * y as f32;
                sm += v;
            }
        }
        let (cx, cy) = if sm > 0.0 { (sx / sm, sy / sm) } else { (tw as f32 / 2.0, th as f32 / 2.0) };
        let ox = ((14.0 - cx).round() as isize).clamp(0, 28 - tw as isize);
        let oy = ((14.0 - cy).round() as isize).clamp(0, 28 - th as isize);
        for y in 0..th {
            for x in 0..tw {
                out[(oy as usize + y) * 28 + ox as usize + x] = (small[y * tw + x].clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
        out
    }
}
