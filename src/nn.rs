//! Das neuronale Netz: Schichten mit Vorwärts- und Rückwärtsrechnung
//! (Backpropagation), Verlustfunktionen und Optimierer.
//!
//! Alles ist bewusst ohne ML-Bibliothek geschrieben. Gerechnet wird in f32,
//! batch-weise. Matrixprodukte laufen über `matrixmultiply` (reines Rust, SIMD),
//! parallelisiert über rayon. Ein Batch liegt zeilenweise im Speicher:
//! Sample i belegt `x[i*len .. (i+1)*len]`, räumliche Daten als (C, H, W).

use crate::rng::Rng;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Hilfsfunktionen
// ---------------------------------------------------------------------------

/// C(m×n) = A(m×k)·B(k×n) + beta·C mit beliebigen Strides (Zeile, Spalte).
#[allow(clippy::too_many_arguments)]
#[inline]
fn gemm(m: usize, k: usize, n: usize, a: &[f32], sa: (usize, usize), b: &[f32], sb: (usize, usize), beta: f32, c: &mut [f32], sc: (usize, usize)) {
    if m == 0 || n == 0 {
        return;
    }
    debug_assert!(k == 0 || a.len() > (m - 1) * sa.0 + (k - 1) * sa.1);
    debug_assert!(k == 0 || b.len() > (k - 1) * sb.0 + (n - 1) * sb.1);
    debug_assert!(c.len() > (m - 1) * sc.0 + (n - 1) * sc.1);
    unsafe {
        matrixmultiply::sgemm(m, k, n, 1.0, a.as_ptr(), sa.0 as isize, sa.1 as isize, b.as_ptr(), sb.0 as isize, sb.1 as isize, beta, c.as_mut_ptr(), sc.0 as isize, sc.1 as isize);
    }
}

/// Teilt `out` (rows × row_len) in Zeilenblöcke auf und bearbeitet sie parallel.
/// `f(erste_zeile, block)`.
fn par_rows(out: &mut [f32], row_len: usize, rows: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let nt = rayon::current_num_threads().max(1);
    if nt == 1 || rows < 2 || rows * row_len < 8192 {
        f(0, out);
        return;
    }
    let chunk = rows.div_ceil(nt).max(1);
    out.par_chunks_mut(chunk * row_len).enumerate().for_each(|(i, c)| f(i * chunk, c));
}

fn use_par(b: usize) -> bool {
    b >= 4 && rayon::current_num_threads() > 1
}

// ---------------------------------------------------------------------------
// Formen, Parameter, Initialisierung
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl Shape {
    pub fn new(c: usize, h: usize, w: usize) -> Self {
        Self { c, h, w }
    }
    pub fn flat(n: usize) -> Self {
        Self { c: n, h: 1, w: 1 }
    }
    pub fn len(&self) -> usize {
        self.c * self.h * self.w
    }
    pub fn spatial(&self) -> bool {
        self.h > 1 || self.w > 1
    }
    pub fn describe(&self) -> String {
        if self.spatial() { format!("{}×{}×{}", self.c, self.h, self.w) } else { format!("{}", self.c) }
    }
}

/// Ein Parametertensor samt Gradient und Optimiererzustand.
#[derive(Clone, Default)]
pub struct Param {
    pub w: Vec<f32>,
    pub g: Vec<f32>,
    /// Gewichtsabnahme (Weight Decay) anwenden? Nein für Bias und BatchNorm.
    pub decay: bool,
    /// Erstes/zweites Moment (Momentum, Adam, RMSProp).
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl Param {
    fn new(w: Vec<f32>, decay: bool) -> Self {
        let n = w.len();
        Self { w, g: vec![0.0; n], decay, m: Vec::new(), v: Vec::new() }
    }
    /// Kopie mit Werten und letztem Gradienten, ohne Optimiererzustand.
    fn fresh(&self) -> Self {
        Self { w: self.w.clone(), g: self.g.clone(), decay: self.decay, m: Vec::new(), v: Vec::new() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Init {
    He,
    Xavier,
    LeCun,
    Small,
    Zero,
}

impl Init {
    pub const ALL: [Init; 5] = [Init::He, Init::Xavier, Init::LeCun, Init::Small, Init::Zero];
    pub fn name(self) -> &'static str {
        match self {
            Init::He => "He (Kaiming) normal",
            Init::Xavier => "Xavier/Glorot uniform",
            Init::LeCun => "LeCun normal",
            Init::Small => "Normal σ=0,01",
            Init::Zero => "Null (Symmetrie!)",
        }
    }
    fn fill(self, n: usize, fan_in: usize, fan_out: usize, rng: &mut Rng) -> Vec<f32> {
        let (fi, fo) = (fan_in as f32, fan_out as f32);
        (0..n)
            .map(|_| match self {
                Init::He => rng.normal() * (2.0 / fi).sqrt(),
                Init::LeCun => rng.normal() * (1.0 / fi).sqrt(),
                Init::Xavier => (rng.uniform() * 2.0 - 1.0) * (6.0 / (fi + fo)).sqrt(),
                Init::Small => rng.normal() * 0.01,
                Init::Zero => 0.0,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Schichten
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActKind {
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    GELU,
}

impl ActKind {
    pub const ALL: [ActKind; 5] = [ActKind::ReLU, ActKind::LeakyReLU, ActKind::Sigmoid, ActKind::Tanh, ActKind::GELU];
    pub fn name(self) -> &'static str {
        match self {
            ActKind::ReLU => "ReLU",
            ActKind::LeakyReLU => "LeakyReLU",
            ActKind::Sigmoid => "Sigmoid",
            ActKind::Tanh => "Tanh",
            ActKind::GELU => "GELU",
        }
    }
    #[inline]
    fn f(self, x: f32) -> f32 {
        match self {
            ActKind::ReLU => x.max(0.0),
            ActKind::LeakyReLU => if x > 0.0 { x } else { 0.01 * x },
            ActKind::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActKind::Tanh => x.tanh(),
            ActKind::GELU => {
                let u = 0.797_884_6 * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + u.tanh())
            }
        }
    }
    /// Ableitung, gegeben Eingabe x und Ausgabe y.
    #[inline]
    fn df(self, x: f32, y: f32) -> f32 {
        match self {
            ActKind::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActKind::LeakyReLU => if x > 0.0 { 1.0 } else { 0.01 },
            ActKind::Sigmoid => y * (1.0 - y),
            ActKind::Tanh => 1.0 - y * y,
            ActKind::GELU => {
                let c = 0.797_884_6;
                let t = (c * (x + 0.044715 * x * x * x)).tanh();
                0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * x * x)
            }
        }
    }
    /// Anteil "toter" (ReLU = 0) bzw. gesättigter (Sigmoid/Tanh) Einheiten.
    fn is_dead(self, x: f32, y: f32) -> bool {
        match self {
            ActKind::ReLU | ActKind::GELU => x <= 0.0,
            ActKind::LeakyReLU => x <= 0.0,
            ActKind::Sigmoid => !(0.02..=0.98).contains(&y),
            ActKind::Tanh => y.abs() > 0.96,
        }
    }
}

#[derive(Clone)]
pub struct Dense {
    pub inp: usize,
    pub out: usize,
    pub w: Param, // (out × inp)
    pub b: Param, // (out)
    x: Vec<f32>,
}

#[derive(Clone)]
pub struct Conv {
    pub cin: usize,
    pub h: usize,
    pub wd: usize,
    pub cout: usize,
    pub k: usize,
    pub pad: usize,
    pub ho: usize,
    pub wo: usize,
    pub w: Param, // (cout × cin·k·k)
    pub b: Param, // (cout)
    cols: Vec<f32>,
}

#[derive(Clone)]
pub struct MaxPool {
    pub size: usize,
    pub inp: Shape,
    pub out: Shape,
    arg: Vec<u32>,
}

#[derive(Clone)]
pub struct Act {
    pub kind: ActKind,
    pub shape: Shape,
    x: Vec<f32>,
    y: Vec<f32>,
}

#[derive(Clone)]
pub struct Dropout {
    pub p: f32,
    pub shape: Shape,
    mask: Vec<f32>,
    rng: Rng,
}

#[derive(Clone)]
pub struct BatchNorm {
    pub shape: Shape,
    pub gamma: Param,
    pub beta: Param,
    pub run_mean: Vec<f32>,
    pub run_var: Vec<f32>,
    xhat: Vec<f32>,
    inv_std: Vec<f32>,
}

#[derive(Clone)]
pub enum Layer {
    Dense(Dense),
    Conv(Conv),
    Pool(MaxPool),
    Act(Act),
    Dropout(Dropout),
    BatchNorm(BatchNorm),
}

impl Layer {
    pub fn name(&self) -> String {
        match self {
            Layer::Dense(d) => format!("Dense {}", d.out),
            Layer::Conv(c) => format!("Conv {}× {}×{}{}", c.cout, c.k, c.k, if c.pad > 0 { " same" } else { "" }),
            Layer::Pool(p) => format!("MaxPool {}×{}", p.size, p.size),
            Layer::Act(a) => a.kind.name().to_string(),
            Layer::Dropout(d) => format!("Dropout {:.2}", d.p),
            Layer::BatchNorm(_) => "BatchNorm".to_string(),
        }
    }

    pub fn out_shape(&self) -> Shape {
        match self {
            Layer::Dense(d) => Shape::flat(d.out),
            Layer::Conv(c) => Shape::new(c.cout, c.ho, c.wo),
            Layer::Pool(p) => p.out,
            Layer::Act(a) => a.shape,
            Layer::Dropout(d) => d.shape,
            Layer::BatchNorm(b) => b.shape,
        }
    }

    pub fn has_params(&self) -> bool {
        matches!(self, Layer::Dense(_) | Layer::Conv(_) | Layer::BatchNorm(_))
    }

    /// (Gewichte, Bias) bzw. (γ, β).
    pub fn params(&self) -> Vec<&Param> {
        match self {
            Layer::Dense(d) => vec![&d.w, &d.b],
            Layer::Conv(c) => vec![&c.w, &c.b],
            Layer::BatchNorm(b) => vec![&b.gamma, &b.beta],
            _ => vec![],
        }
    }

    fn params_mut(&mut self) -> Vec<&mut Param> {
        match self {
            Layer::Dense(d) => vec![&mut d.w, &mut d.b],
            Layer::Conv(c) => vec![&mut c.w, &mut c.b],
            Layer::BatchNorm(b) => vec![&mut b.gamma, &mut b.beta],
            _ => vec![],
        }
    }

    pub fn n_params(&self) -> usize {
        self.params().iter().map(|p| p.w.len()).sum()
    }

    /// Multiply-Accumulate-Operationen pro Sample im Vorwärtslauf.
    pub fn macs(&self) -> usize {
        match self {
            Layer::Dense(d) => d.inp * d.out,
            Layer::Conv(c) => c.cout * c.cin * c.k * c.k * c.ho * c.wo,
            _ => 0,
        }
    }

    /// Kopie nur mit Parametern (ohne Zwischenspeicher und Optimiererzustand).
    fn fresh(&self) -> Layer {
        match self {
            Layer::Dense(d) => Layer::Dense(Dense { inp: d.inp, out: d.out, w: d.w.fresh(), b: d.b.fresh(), x: Vec::new() }),
            Layer::Conv(c) => Layer::Conv(Conv { cin: c.cin, h: c.h, wd: c.wd, cout: c.cout, k: c.k, pad: c.pad, ho: c.ho, wo: c.wo, w: c.w.fresh(), b: c.b.fresh(), cols: Vec::new() }),
            Layer::Pool(p) => Layer::Pool(MaxPool { size: p.size, inp: p.inp, out: p.out, arg: Vec::new() }),
            Layer::Act(a) => Layer::Act(Act { kind: a.kind, shape: a.shape, x: Vec::new(), y: Vec::new() }),
            Layer::Dropout(d) => Layer::Dropout(Dropout { p: d.p, shape: d.shape, mask: Vec::new(), rng: d.rng.clone() }),
            Layer::BatchNorm(b) => Layer::BatchNorm(BatchNorm {
                shape: b.shape,
                gamma: b.gamma.fresh(),
                beta: b.beta.fresh(),
                run_mean: b.run_mean.clone(),
                run_var: b.run_var.clone(),
                xhat: Vec::new(),
                inv_std: Vec::new(),
            }),
        }
    }

    fn forward(&mut self, x: &[f32], b: usize, train: bool) -> Vec<f32> {
        match self {
            Layer::Dense(l) => l.forward(x, b, train),
            Layer::Conv(l) => l.forward(x, b, train),
            Layer::Pool(l) => l.forward(x, b, train),
            Layer::Act(l) => l.forward(x, train),
            Layer::Dropout(l) => l.forward(x, train),
            Layer::BatchNorm(l) => l.forward(x, b, train),
        }
    }

    fn backward(&mut self, dy: &[f32], b: usize, need_dx: bool) -> Vec<f32> {
        match self {
            Layer::Dense(l) => l.backward(dy, b, need_dx),
            Layer::Conv(l) => l.backward(dy, b, need_dx),
            Layer::Pool(l) => l.backward(dy, b),
            Layer::Act(l) => l.backward(dy),
            Layer::Dropout(l) => l.backward(dy),
            Layer::BatchNorm(l) => l.backward(dy, b),
        }
    }
}

// --- Dense ------------------------------------------------------------------

impl Dense {
    fn new(inp: usize, out: usize, init: Init, rng: &mut Rng) -> Self {
        Self { inp, out, w: Param::new(init.fill(inp * out, inp, out, rng), true), b: Param::new(vec![0.0; out], false), x: Vec::new() }
    }

    fn forward(&mut self, x: &[f32], b: usize, train: bool) -> Vec<f32> {
        let (i, o) = (self.inp, self.out);
        let mut y = vec![0.0f32; b * o];
        let w = &self.w.w;
        let bias = &self.b.w;
        // y = x · Wᵀ + b
        par_rows(&mut y, o, b, |r0, yc| {
            let rows = yc.len() / o;
            gemm(rows, i, o, &x[r0 * i..], (i, 1), w, (1, i), 0.0, yc, (o, 1));
            for r in yc.chunks_mut(o) {
                for (v, bb) in r.iter_mut().zip(bias) {
                    *v += bb;
                }
            }
        });
        if train {
            self.x.clear();
            self.x.extend_from_slice(x);
        }
        y
    }

    fn backward(&mut self, dy: &[f32], b: usize, need_dx: bool) -> Vec<f32> {
        let (i, o) = (self.inp, self.out);
        let x = &self.x;
        // dW = dyᵀ · x   (o × i), parallel über Ausgabezeilen
        par_rows(&mut self.w.g, i, o, |r0, gc| {
            let rows = gc.len() / i;
            gemm(rows, b, i, &dy[r0..], (1, o), x, (i, 1), 0.0, gc, (i, 1));
        });
        // db = Σ_batch dy
        let db = &mut self.b.g;
        db.iter_mut().for_each(|v| *v = 0.0);
        for r in dy.chunks(o) {
            for (g, d) in db.iter_mut().zip(r) {
                *g += d;
            }
        }
        if !need_dx {
            return Vec::new();
        }
        // dx = dy · W   (b × i)
        let mut dx = vec![0.0f32; b * i];
        let w = &self.w.w;
        par_rows(&mut dx, i, b, |r0, dc| {
            let rows = dc.len() / i;
            gemm(rows, o, i, &dy[r0 * o..], (o, 1), w, (i, 1), 0.0, dc, (i, 1));
        });
        dx
    }
}

// --- Conv2D (im2col) ----------------------------------------------------------

impl Conv {
    fn new(inp: Shape, cout: usize, k: usize, same: bool, init: Init, rng: &mut Rng) -> Result<Self, String> {
        let pad = if same { k / 2 } else { 0 };
        if inp.h + 2 * pad < k || inp.w + 2 * pad < k {
            return Err(format!("Kernel {k}×{k} ist größer als die Eingabe {}", inp.describe()));
        }
        let ho = inp.h + 2 * pad - k + 1;
        let wo = inp.w + 2 * pad - k + 1;
        let ckk = inp.c * k * k;
        Ok(Self {
            cin: inp.c,
            h: inp.h,
            wd: inp.w,
            cout,
            k,
            pad,
            ho,
            wo,
            w: Param::new(init.fill(cout * ckk, ckk, cout * k * k, rng), true),
            b: Param::new(vec![0.0; cout], false),
            cols: Vec::new(),
        })
    }

    fn ckk(&self) -> usize {
        self.cin * self.k * self.k
    }

    fn im2col(&self, x: &[f32], col: &mut [f32]) {
        let (k, p, ho, wo, h, w) = (self.k, self.pad as isize, self.ho, self.wo, self.h as isize, self.wd as isize);
        let hw = ho * wo;
        for c in 0..self.cin {
            let xc = &x[c * self.h * self.wd..];
            for ky in 0..k {
                for kx in 0..k {
                    let row = &mut col[((c * k + ky) * k + kx) * hw..][..hw];
                    for oy in 0..ho {
                        let iy = oy as isize + ky as isize - p;
                        let r = &mut row[oy * wo..][..wo];
                        if iy < 0 || iy >= h {
                            r.iter_mut().for_each(|v| *v = 0.0);
                            continue;
                        }
                        let xr = &xc[iy as usize * self.wd..];
                        for (ox, v) in r.iter_mut().enumerate() {
                            let ix = ox as isize + kx as isize - p;
                            *v = if ix < 0 || ix >= w { 0.0 } else { xr[ix as usize] };
                        }
                    }
                }
            }
        }
    }

    fn col2im(&self, col: &[f32], dx: &mut [f32]) {
        let (k, p, ho, wo, h, w) = (self.k, self.pad as isize, self.ho, self.wo, self.h as isize, self.wd as isize);
        let hw = ho * wo;
        dx.iter_mut().for_each(|v| *v = 0.0);
        for c in 0..self.cin {
            for ky in 0..k {
                for kx in 0..k {
                    let row = &col[((c * k + ky) * k + kx) * hw..][..hw];
                    for oy in 0..ho {
                        let iy = oy as isize + ky as isize - p;
                        if iy < 0 || iy >= h {
                            continue;
                        }
                        let base = c * self.h * self.wd + iy as usize * self.wd;
                        for ox in 0..wo {
                            let ix = ox as isize + kx as isize - p;
                            if ix >= 0 && ix < w {
                                dx[base + ix as usize] += row[oy * wo + ox];
                            }
                        }
                    }
                }
            }
        }
    }

    fn forward(&mut self, x: &[f32], b: usize, train: bool) -> Vec<f32> {
        let (ckk, hw, cout) = (self.ckk(), self.ho * self.wo, self.cout);
        let in_len = self.cin * self.h * self.wd;
        let out_len = cout * hw;
        let mut y = vec![0.0f32; b * out_len];
        let mut cols = std::mem::take(&mut self.cols);
        cols.resize(b * ckk * hw, 0.0);
        let this = &*self;
        let one = |s: usize, ys: &mut [f32], col: &mut [f32]| {
            this.im2col(&x[s * in_len..][..in_len], col);
            gemm(cout, ckk, hw, &this.w.w, (ckk, 1), col, (hw, 1), 0.0, ys, (hw, 1));
            for (c, r) in ys.chunks_mut(hw).enumerate() {
                let bb = this.b.w[c];
                r.iter_mut().for_each(|v| *v += bb);
            }
        };
        if use_par(b) {
            y.par_chunks_mut(out_len).zip(cols.par_chunks_mut(ckk * hw)).enumerate().for_each(|(s, (ys, col))| one(s, ys, col));
        } else {
            for (s, (ys, col)) in y.chunks_mut(out_len).zip(cols.chunks_mut(ckk * hw)).enumerate() {
                one(s, ys, col);
            }
        }
        if train {
            self.cols = cols;
        }
        y
    }

    fn backward(&mut self, dy: &[f32], b: usize, need_dx: bool) -> Vec<f32> {
        let (ckk, hw, cout) = (self.ckk(), self.ho * self.wo, self.cout);
        let out_len = cout * hw;
        let in_len = self.cin * self.h * self.wd;
        let wlen = cout * ckk;
        let cols = &self.cols;
        // dW = Σ_s dy_s · col_sᵀ ; db = Σ_s Σ_hw dy_s
        let acc = |mut a: Vec<f32>, s: usize| {
            let dys = &dy[s * out_len..][..out_len];
            gemm(cout, hw, ckk, dys, (hw, 1), &cols[s * ckk * hw..], (1, hw), 1.0, &mut a[..wlen], (ckk, 1));
            for (c, r) in dys.chunks(hw).enumerate() {
                a[wlen + c] += r.iter().sum::<f32>();
            }
            a
        };
        let grads = if use_par(b) {
            (0..b).into_par_iter().fold(|| vec![0.0f32; wlen + cout], acc).reduce(
                || vec![0.0f32; wlen + cout],
                |mut a, c| {
                    a.iter_mut().zip(&c).for_each(|(x, y)| *x += y);
                    a
                },
            )
        } else {
            (0..b).fold(vec![0.0f32; wlen + cout], acc)
        };
        self.w.g.copy_from_slice(&grads[..wlen]);
        self.b.g.copy_from_slice(&grads[wlen..]);
        if !need_dx {
            return Vec::new();
        }
        // dcol = Wᵀ · dy_s ; dx_s = col2im(dcol)
        let mut dx = vec![0.0f32; b * in_len];
        let this = &*self;
        let one = |s: usize, dxs: &mut [f32], dcol: &mut Vec<f32>| {
            dcol.resize(ckk * hw, 0.0);
            gemm(ckk, cout, hw, &this.w.w, (1, ckk), &dy[s * out_len..], (hw, 1), 0.0, dcol, (hw, 1));
            this.col2im(dcol, dxs);
        };
        if use_par(b) {
            dx.par_chunks_mut(in_len).enumerate().for_each_init(Vec::new, |dcol, (s, dxs)| one(s, dxs, dcol));
        } else {
            let mut dcol = Vec::new();
            for (s, dxs) in dx.chunks_mut(in_len).enumerate() {
                one(s, dxs, &mut dcol);
            }
        }
        dx
    }
}

// --- MaxPool ------------------------------------------------------------------

impl MaxPool {
    fn new(inp: Shape, size: usize) -> Result<Self, String> {
        if inp.h < size || inp.w < size {
            return Err(format!("Pooling {size}×{size} passt nicht auf {}", inp.describe()));
        }
        Ok(Self { size, inp, out: Shape::new(inp.c, inp.h / size, inp.w / size), arg: Vec::new() })
    }

    fn forward(&mut self, x: &[f32], b: usize, train: bool) -> Vec<f32> {
        let (s, i, o) = (self.size, self.inp, self.out);
        let (il, ol) = (i.len(), o.len());
        let mut y = vec![0.0f32; b * ol];
        let mut arg = vec![0u32; b * ol];
        for n in 0..b {
            let xs = &x[n * il..][..il];
            for c in 0..o.c {
                for oy in 0..o.h {
                    for ox in 0..o.w {
                        let mut best = f32::NEG_INFINITY;
                        let mut bi = 0usize;
                        for dy in 0..s {
                            for dx in 0..s {
                                let idx = c * i.h * i.w + (oy * s + dy) * i.w + ox * s + dx;
                                if xs[idx] > best {
                                    best = xs[idx];
                                    bi = idx;
                                }
                            }
                        }
                        let oi = n * ol + c * o.h * o.w + oy * o.w + ox;
                        y[oi] = best;
                        arg[oi] = bi as u32;
                    }
                }
            }
        }
        if train {
            self.arg = arg;
        }
        y
    }

    fn backward(&mut self, dy: &[f32], b: usize) -> Vec<f32> {
        let (il, ol) = (self.inp.len(), self.out.len());
        let mut dx = vec![0.0f32; b * il];
        for n in 0..b {
            for j in 0..ol {
                dx[n * il + self.arg[n * ol + j] as usize] += dy[n * ol + j];
            }
        }
        dx
    }
}

// --- Aktivierung --------------------------------------------------------------

impl Act {
    fn forward(&mut self, x: &[f32], train: bool) -> Vec<f32> {
        let k = self.kind;
        let y: Vec<f32> = if x.len() > 1 << 16 { x.par_iter().map(|&v| k.f(v)).collect() } else { x.iter().map(|&v| k.f(v)).collect() };
        if train {
            self.x.clear();
            self.x.extend_from_slice(x);
            self.y.clear();
            self.y.extend_from_slice(&y);
        }
        y
    }

    fn backward(&mut self, dy: &[f32]) -> Vec<f32> {
        let k = self.kind;
        dy.iter().zip(&self.x).zip(&self.y).map(|((d, &x), &y)| d * k.df(x, y)).collect()
    }
}

// --- Dropout (inverted) -------------------------------------------------------

impl Dropout {
    fn forward(&mut self, x: &[f32], train: bool) -> Vec<f32> {
        if !train || self.p <= 0.0 {
            self.mask.clear();
            return x.to_vec();
        }
        let keep = 1.0 - self.p;
        self.mask = (0..x.len()).map(|_| if self.rng.uniform() < keep { 1.0 / keep } else { 0.0 }).collect();
        x.iter().zip(&self.mask).map(|(a, m)| a * m).collect()
    }

    fn backward(&mut self, dy: &[f32]) -> Vec<f32> {
        if self.mask.is_empty() {
            return dy.to_vec();
        }
        dy.iter().zip(&self.mask).map(|(a, m)| a * m).collect()
    }
}

// --- BatchNorm (pro Kanal bzw. pro Neuron) ------------------------------------

impl BatchNorm {
    const EPS: f32 = 1e-5;
    const MOM: f32 = 0.1;

    fn new(shape: Shape) -> Self {
        let c = shape.c;
        Self {
            shape,
            gamma: Param::new(vec![1.0; c], false),
            beta: Param::new(vec![0.0; c], false),
            run_mean: vec![0.0; c],
            run_var: vec![1.0; c],
            xhat: Vec::new(),
            inv_std: Vec::new(),
        }
    }

    fn forward(&mut self, x: &[f32], b: usize, train: bool) -> Vec<f32> {
        let (c, s) = (self.shape.c, self.shape.h * self.shape.w);
        let len = c * s;
        let mut y = vec![0.0f32; x.len()];
        if !train {
            for n in 0..b {
                for ch in 0..c {
                    let inv = 1.0 / (self.run_var[ch] + Self::EPS).sqrt();
                    let (g, be, mu) = (self.gamma.w[ch], self.beta.w[ch], self.run_mean[ch]);
                    for j in 0..s {
                        let i = n * len + ch * s + j;
                        y[i] = g * (x[i] - mu) * inv + be;
                    }
                }
            }
            return y;
        }
        let m = (b * s) as f32;
        let mut xhat = vec![0.0f32; x.len()];
        let mut inv_std = vec![0.0f32; c];
        for ch in 0..c {
            let mut sum = 0.0f64;
            for n in 0..b {
                sum += x[n * len + ch * s..][..s].iter().map(|&v| v as f64).sum::<f64>();
            }
            let mu = (sum / m as f64) as f32;
            let mut var = 0.0f64;
            for n in 0..b {
                var += x[n * len + ch * s..][..s].iter().map(|&v| ((v - mu) as f64).powi(2)).sum::<f64>();
            }
            let var = (var / m as f64) as f32;
            let inv = 1.0 / (var + Self::EPS).sqrt();
            inv_std[ch] = inv;
            let (g, be) = (self.gamma.w[ch], self.beta.w[ch]);
            for n in 0..b {
                for j in 0..s {
                    let i = n * len + ch * s + j;
                    let xh = (x[i] - mu) * inv;
                    xhat[i] = xh;
                    y[i] = g * xh + be;
                }
            }
            self.run_mean[ch] = (1.0 - Self::MOM) * self.run_mean[ch] + Self::MOM * mu;
            let unbiased = if m > 1.0 { var * m / (m - 1.0) } else { var };
            self.run_var[ch] = (1.0 - Self::MOM) * self.run_var[ch] + Self::MOM * unbiased;
        }
        self.xhat = xhat;
        self.inv_std = inv_std;
        y
    }

    fn backward(&mut self, dy: &[f32], b: usize) -> Vec<f32> {
        let (c, s) = (self.shape.c, self.shape.h * self.shape.w);
        let len = c * s;
        let m = (b * s) as f32;
        let mut dx = vec![0.0f32; dy.len()];
        for ch in 0..c {
            let (mut sdy, mut sdyx) = (0.0f32, 0.0f32);
            for n in 0..b {
                for j in 0..s {
                    let i = n * len + ch * s + j;
                    sdy += dy[i];
                    sdyx += dy[i] * self.xhat[i];
                }
            }
            self.gamma.g[ch] = sdyx;
            self.beta.g[ch] = sdy;
            let g = self.gamma.w[ch];
            let k = g * self.inv_std[ch] / m;
            for n in 0..b {
                for j in 0..s {
                    let i = n * len + ch * s + j;
                    dx[i] = k * (m * dy[i] - sdy - self.xhat[i] * sdyx);
                }
            }
        }
        dx
    }
}

// ---------------------------------------------------------------------------
// Architektur-Beschreibung und Modell
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LayerSpec {
    Dense { units: usize },
    Conv { filters: usize, k: usize, same: bool },
    MaxPool { size: usize },
    Act(ActKind),
    Dropout { p: f32 },
    BatchNorm,
}

impl LayerSpec {
    pub fn label(&self) -> String {
        match *self {
            LayerSpec::Dense { units } => format!("Dense {units}"),
            LayerSpec::Conv { filters, k, same } => format!("Conv {filters}× {k}×{k}{}", if same { " same" } else { "" }),
            LayerSpec::MaxPool { size } => format!("MaxPool {size}×{size}"),
            LayerSpec::Act(a) => a.name().to_string(),
            LayerSpec::Dropout { p } => format!("Dropout {p:.2}"),
            LayerSpec::BatchNorm => "BatchNorm".into(),
        }
    }
}

#[derive(Clone)]
pub struct Model {
    pub input: Shape,
    pub classes: usize,
    pub layers: Vec<Layer>,
}

/// Statistik der Ausgabe einer Schicht für einen Trainingsbatch.
#[derive(Clone, Copy, Default, Debug)]
pub struct ActStat {
    pub mean: f32,
    pub std: f32,
    /// Bei Aktivierungen: Anteil toter/gesättigter Einheiten.
    pub dead: f32,
}

impl Model {
    /// Baut das Netz aus der Beschreibung. Am Ende wird immer eine
    /// Ausgabeschicht Dense(classes) angehängt (Softmax steckt in der Verlustfunktion).
    pub fn build(specs: &[LayerSpec], input: Shape, classes: usize, init: Init, seed: u64) -> Result<Model, String> {
        let mut rng = Rng::new(seed);
        let mut shape = input;
        let mut layers = Vec::new();
        for (i, s) in specs.iter().enumerate() {
            let err = |e: String| format!("Schicht {} ({}): {e}", i + 1, s.label());
            let l = match *s {
                LayerSpec::Dense { units } => {
                    if units == 0 {
                        return Err(err("0 Neuronen".into()));
                    }
                    Layer::Dense(Dense::new(shape.len(), units, init, &mut rng))
                }
                LayerSpec::Conv { filters, k, same } => {
                    if !shape.spatial() {
                        return Err(err("Conv braucht eine räumliche Eingabe. Nach einer Dense-Schicht ist das Bild flach.".into()));
                    }
                    Layer::Conv(Conv::new(shape, filters.max(1), k.max(1), same, init, &mut rng).map_err(err)?)
                }
                LayerSpec::MaxPool { size } => {
                    if !shape.spatial() {
                        return Err(err("Pooling braucht eine räumliche Eingabe.".into()));
                    }
                    Layer::Pool(MaxPool::new(shape, size.max(2)).map_err(err)?)
                }
                LayerSpec::Act(kind) => Layer::Act(Act { kind, shape, x: Vec::new(), y: Vec::new() }),
                LayerSpec::Dropout { p } => Layer::Dropout(Dropout { p: p.clamp(0.0, 0.95), shape, mask: Vec::new(), rng: Rng::new(rng.next_u64()) }),
                LayerSpec::BatchNorm => Layer::BatchNorm(BatchNorm::new(shape)),
            };
            shape = l.out_shape();
            layers.push(l);
        }
        layers.push(Layer::Dense(Dense::new(shape.len(), classes, init, &mut rng)));
        Ok(Model { input, classes, layers })
    }

    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }

    pub fn macs(&self) -> usize {
        self.layers.iter().map(|l| l.macs()).sum()
    }

    /// Kopie für die Anzeige (nur Parameter, keine Zwischenspeicher).
    pub fn snapshot(&self) -> Model {
        Model { input: self.input, classes: self.classes, layers: self.layers.iter().map(|l| l.fresh()).collect() }
    }

    /// Vorwärtslauf; liefert die Logits (b × classes).
    pub fn forward(&mut self, x: &[f32], b: usize, train: bool, mut stats: Option<&mut Vec<ActStat>>) -> Vec<f32> {
        if let Some(s) = stats.as_deref_mut() {
            s.clear();
        }
        let mut cur = x.to_vec();
        for l in &mut self.layers {
            let y = l.forward(&cur, b, train);
            if let Some(s) = stats.as_deref_mut() {
                s.push(layer_stat(l, &cur, &y));
            }
            cur = y;
        }
        cur
    }

    /// Vorwärtslauf für ein einzelnes Sample; liefert die Ausgabe jeder Schicht.
    pub fn forward_trace(&mut self, x: &[f32]) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(self.layers.len());
        let mut cur = x.to_vec();
        for l in &mut self.layers {
            cur = l.forward(&cur, 1, false);
            out.push(cur.clone());
        }
        out
    }

    /// Rückwärtslauf; füllt die Gradienten aller Parameter.
    pub fn backward(&mut self, dlogits: &[f32], b: usize) {
        let mut d = dlogits.to_vec();
        for (i, l) in self.layers.iter_mut().enumerate().rev() {
            d = l.backward(&d, b, i > 0);
        }
    }

    /// Alle Parameter mit dem Index ihrer Schicht.
    pub fn params_mut(&mut self) -> Vec<(usize, &mut Param)> {
        self.layers.iter_mut().enumerate().flat_map(|(i, l)| l.params_mut().into_iter().map(move |p| (i, p))).collect()
    }
}

fn layer_stat(l: &Layer, x: &[f32], y: &[f32]) -> ActStat {
    let n = y.len().max(1) as f64;
    let mean = y.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = y.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let dead = if let Layer::Act(a) = l { x.iter().zip(y).filter(|(a2, b2)| a.kind.is_dead(**a2, **b2)).count() as f64 / n } else { 0.0 };
    ActStat { mean: mean as f32, std: var.sqrt() as f32, dead: dead as f32 }
}

// ---------------------------------------------------------------------------
// Verlustfunktionen
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossKind {
    CrossEntropy,
    Mse,
}

impl LossKind {
    pub fn name(self) -> &'static str {
        match self {
            LossKind::CrossEntropy => "Softmax + Kreuzentropie",
            LossKind::Mse => "Softmax + MSE",
        }
    }
}

pub fn softmax(logits: &[f32], out: &mut [f32]) {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for (o, &l) in out.iter_mut().zip(logits) {
        *o = (l - mx).exp();
        s += *o;
    }
    out.iter_mut().for_each(|o| *o /= s);
}

/// Liefert (mittlerer Verlust, Anzahl richtig, dLoss/dLogits).
pub fn loss_grad(logits: &[f32], labels: &[u8], k: usize, kind: LossKind, smoothing: f32) -> (f32, usize, Vec<f32>) {
    let b = labels.len();
    let mut grad = vec![0.0f32; logits.len()];
    let mut loss = 0.0f64;
    let mut correct = 0;
    let mut p = vec![0.0f32; k];
    let (on, off) = (1.0 - smoothing + smoothing / k as f32, smoothing / k as f32);
    for n in 0..b {
        let z = &logits[n * k..][..k];
        softmax(z, &mut p);
        let y = labels[n] as usize;
        let pred = (0..k).max_by(|&a, &c| z[a].total_cmp(&z[c])).unwrap_or(0);
        if pred == y {
            correct += 1;
        }
        let g = &mut grad[n * k..][..k];
        match kind {
            LossKind::CrossEntropy => {
                for j in 0..k {
                    let t = if j == y { on } else { off };
                    if t > 0.0 {
                        loss -= (t * p[j].max(1e-12).ln()) as f64;
                    }
                    g[j] = (p[j] - t) / b as f32;
                }
            }
            LossKind::Mse => {
                // L = Σ (p - t)², Gradient durch die Softmax-Jacobimatrix
                let dp: Vec<f32> = (0..k).map(|j| 2.0 * (p[j] - if j == y { on } else { off })).collect();
                loss += (0..k).map(|j| ((p[j] - if j == y { on } else { off }) as f64).powi(2)).sum::<f64>();
                let dot: f32 = (0..k).map(|j| dp[j] * p[j]).sum();
                for j in 0..k {
                    g[j] = p[j] * (dp[j] - dot) / b as f32;
                }
            }
        }
    }
    ((loss / b as f64) as f32, correct, grad)
}

// ---------------------------------------------------------------------------
// Optimierer
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptKind {
    Sgd,
    Momentum,
    Nesterov,
    RmsProp,
    Adam,
    AdamW,
}

impl OptKind {
    pub const ALL: [OptKind; 6] = [OptKind::Sgd, OptKind::Momentum, OptKind::Nesterov, OptKind::RmsProp, OptKind::Adam, OptKind::AdamW];
    pub fn name(self) -> &'static str {
        match self {
            OptKind::Sgd => "SGD",
            OptKind::Momentum => "SGD + Momentum",
            OptKind::Nesterov => "SGD + Nesterov",
            OptKind::RmsProp => "RMSProp",
            OptKind::Adam => "Adam",
            OptKind::AdamW => "AdamW",
        }
    }
    pub fn default_lr(self) -> f32 {
        match self {
            OptKind::Sgd | OptKind::Momentum | OptKind::Nesterov => 0.05,
            OptKind::RmsProp | OptKind::Adam | OptKind::AdamW => 0.001,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OptCfg {
    pub kind: OptKind,
    pub momentum: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

/// Pro Parameterschicht: (‖∇W‖, ‖ΔW‖ / ‖W‖).
pub struct StepStats {
    pub grad_norm: Vec<(usize, f32)>,
    pub update_ratio: Vec<(usize, f32)>,
    pub global_grad_norm: f32,
}

/// Ein Optimierungsschritt. `t` zählt ab 1 (für Adams Bias-Korrektur).
pub fn optimize(model: &mut Model, cfg: &OptCfg, lr: f32, t: u64, clip: Option<f32>) -> StepStats {
    let mut params = model.params_mut();
    let global = params.iter().map(|(_, p)| p.g.iter().map(|g| (g * g) as f64).sum::<f64>()).sum::<f64>().sqrt() as f32;
    let scale = match clip {
        Some(c) if global > c && global > 0.0 => c / global,
        _ => 1.0,
    };
    let mut grad_norm = Vec::new();
    let mut update_ratio = Vec::new();
    for (li, p) in params.iter_mut() {
        let n = p.w.len();
        if p.m.len() != n {
            p.m = vec![0.0; n];
        }
        if matches!(cfg.kind, OptKind::RmsProp | OptKind::Adam | OptKind::AdamW) && p.v.len() != n {
            p.v = vec![0.0; n];
        }
        let wd = if p.decay { cfg.weight_decay } else { 0.0 };
        let (mut gsq, mut usq, mut wsq) = (0.0f64, 0.0f64, 0.0f64);
        let (b1t, b2t) = (1.0 - cfg.beta1.powi(t as i32), 1.0 - cfg.beta2.powi(t as i32));
        for i in 0..n {
            let mut g = p.g[i] * scale;
            gsq += (g * g) as f64;
            let w = p.w[i];
            wsq += (w * w) as f64;
            // L2-Regularisierung (gekoppelt) für alle außer AdamW
            if cfg.kind != OptKind::AdamW {
                g += wd * w;
            }
            let upd = match cfg.kind {
                OptKind::Sgd => lr * g,
                OptKind::Momentum => {
                    p.m[i] = cfg.momentum * p.m[i] + g;
                    lr * p.m[i]
                }
                OptKind::Nesterov => {
                    p.m[i] = cfg.momentum * p.m[i] + g;
                    lr * (g + cfg.momentum * p.m[i])
                }
                OptKind::RmsProp => {
                    p.v[i] = cfg.beta2 * p.v[i] + (1.0 - cfg.beta2) * g * g;
                    lr * g / (p.v[i].sqrt() + cfg.eps)
                }
                OptKind::Adam | OptKind::AdamW => {
                    p.m[i] = cfg.beta1 * p.m[i] + (1.0 - cfg.beta1) * g;
                    p.v[i] = cfg.beta2 * p.v[i] + (1.0 - cfg.beta2) * g * g;
                    let mh = p.m[i] / b1t;
                    let vh = p.v[i] / b2t;
                    let mut u = lr * mh / (vh.sqrt() + cfg.eps);
                    if cfg.kind == OptKind::AdamW {
                        u += lr * wd * w;
                    }
                    u
                }
            };
            p.w[i] = w - upd;
            usq += (upd * upd) as f64;
        }
        if p.decay {
            grad_norm.push((*li, gsq.sqrt() as f32));
            update_ratio.push((*li, (usq.sqrt() / wsq.sqrt().max(1e-12)) as f32));
        }
    }
    StepStats { grad_norm, update_ratio, global_grad_norm: global }
}

// ---------------------------------------------------------------------------
// Tests: Gradienten gegen numerische Ableitung prüfen
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn loss_of(m: &mut Model, x: &[f32], y: &[u8], kind: LossKind) -> f32 {
        let b = y.len();
        let z = m.forward(x, b, true, None);
        loss_grad(&z, y, m.classes, kind, 0.1).0
    }

    fn check(specs: &[LayerSpec], input: Shape, kind: LossKind) {
        check_b(specs, input, kind, 3)
    }

    fn check_b(specs: &[LayerSpec], input: Shape, kind: LossKind, b: usize) {
        let mut rng = Rng::new(7);
        let x: Vec<f32> = (0..b * input.len()).map(|_| rng.normal()).collect();
        let y: Vec<u8> = (0..b).map(|i| (i * 3 % 4) as u8).collect();
        let mut m = Model::build(specs, input, 4, Init::He, 3).unwrap();
        // Dropout deterministisch machen: Masken nur bei Training, hier p=0
        let z = m.forward(&x, b, true, None);
        let (_, _, g) = loss_grad(&z, &y, 4, kind, 0.1);
        m.backward(&g, b);
        let analytic: Vec<Vec<f32>> = m.params_mut().iter().map(|(_, p)| p.g.clone()).collect();
        let h = 1e-2f32;
        let mut worst = 0.0f32;
        for pi in 0..analytic.len() {
            let n = analytic[pi].len();
            for i in (0..n).step_by((n / 7).max(1)) {
                let orig = m.params_mut()[pi].1.w[i];
                m.params_mut()[pi].1.w[i] = orig + h;
                let lp = loss_of(&mut m, &x, &y, kind);
                m.params_mut()[pi].1.w[i] = orig - h;
                let lm = loss_of(&mut m, &x, &y, kind);
                m.params_mut()[pi].1.w[i] = orig;
                let num = (lp - lm) / (2.0 * h);
                let a = analytic[pi][i];
                let rel = (num - a).abs() / (num.abs() + a.abs()).max(1e-3);
                worst = worst.max(rel);
                assert!(rel < 5e-2, "Param {pi}[{i}]: numerisch {num}, analytisch {a}");
            }
        }
        eprintln!("max. relativer Fehler {worst:.2e}");
    }

    #[test]
    fn grad_mlp() {
        check(&[LayerSpec::Dense { units: 6 }, LayerSpec::Act(ActKind::Tanh), LayerSpec::Dense { units: 5 }, LayerSpec::Act(ActKind::Sigmoid)], Shape::new(1, 3, 3), LossKind::CrossEntropy);
    }

    #[test]
    fn grad_mse_gelu() {
        check(&[LayerSpec::Dense { units: 6 }, LayerSpec::Act(ActKind::GELU)], Shape::flat(5), LossKind::Mse);
    }

    #[test]
    fn grad_batchnorm() {
        check(&[LayerSpec::Dense { units: 6 }, LayerSpec::BatchNorm, LayerSpec::Act(ActKind::Tanh)], Shape::flat(5), LossKind::CrossEntropy);
    }

    #[test]
    fn grad_cnn() {
        check(
            &[
                LayerSpec::Conv { filters: 3, k: 3, same: true },
                LayerSpec::BatchNorm,
                LayerSpec::Act(ActKind::Tanh),
                LayerSpec::MaxPool { size: 2 },
                LayerSpec::Conv { filters: 2, k: 3, same: false },
                LayerSpec::Act(ActKind::Tanh),
            ],
            Shape::new(2, 6, 6),
            LossKind::CrossEntropy,
        );
    }

    #[test]
    fn grad_cnn_parallel() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(|| {
            check_b(&[LayerSpec::Conv { filters: 3, k: 3, same: true }, LayerSpec::Act(ActKind::Tanh), LayerSpec::Dense { units: 70 }, LayerSpec::Act(ActKind::Tanh)], Shape::new(1, 12, 12), LossKind::CrossEntropy, 64);
        });
    }
}
