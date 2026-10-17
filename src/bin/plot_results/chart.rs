//! A small chart writer: figures made of panels, each panel a pair of axes
//! (linear or log, either may be a list of categories) with the marks the
//! benchmark figures use: bars (grouped, stacked, horizontal), dots, reference
//! lines, whiskers, markers and text, plus legends, titles and axis labels.
//!
//! The output is one self-contained SVG string in the style of `docs/img/*.svg`:
//! an explicit background rect (white; a dark one under
//! `prefers-color-scheme: dark`), the system sans stack, series colours as
//! hex fills. Every mark carries a class (`bar`, `seg`, `dot`, `whisker`, ...),
//! so a reader of the file, or a test, can count what was drawn without
//! parsing geometry. Layout is arithmetic on [`text_width`] estimates, so a
//! figure is a pure function of its inputs: the same data is the same bytes.

use crate::data::R;
use crate::svg::{esc, num, text_width, wrap};

pub const TICK: f64 = 11.0;
pub const LABEL: f64 = 12.0;
pub const TITLE: f64 = 13.0;
pub const HEADLINE: f64 = 15.0;

/// The series palette of the newer benchmark figures.
pub const BLUE: &str = "#2a78d6";
pub const ORANGE: &str = "#eb6834";
pub const GREEN: &str = "#1f9e6e";
pub const GREY: &str = "#8a8985";
pub const GOLD: &str = "#c9a227";
pub const LIGHT_GREY: &str = "#b3b2ac";

/// The muted palette of the older ones (matplotlib's seaborn-muted slots).
pub const M_BLUE: &str = "#4c72b0";
pub const M_ORANGE: &str = "#dd8452";
pub const M_GREEN: &str = "#55a868";
pub const M_RED: &str = "#c44e52";
pub const M_GREY: &str = "#8c8c8c";

const STYLE: &str = "text{font-family:-apple-system,Helvetica,Arial,sans-serif}\n\
.ink{fill:#0b0b0b}.mut{fill:#52514e}.dim{fill:#898781}.bold{font-weight:600}\n\
.bg{fill:#fff}.grid{stroke:#e6e5e1;stroke-width:1}.axis{stroke:#b3b2ac;stroke-width:1}\n\
.sk{stroke:#0b0b0b}.mk{stroke:#52514e}.edge{stroke:#fff}\n\
@media (prefers-color-scheme:dark){.bg{fill:#1a1a19}.ink{fill:#f0efec}.mut{fill:#c3c2b7}.dim{fill:#8c8b84}\n\
.grid{stroke:#2c2c2a}.axis{stroke:#55544f}.sk{stroke:#f0efec}.mk{stroke:#c3c2b7}.edge{stroke:#1a1a19}}";

// ---------------------------------------------------------------------------
// Scales and ticks
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Linear,
    Log,
}

/// A mapping from data values to a 0..1 fraction of an axis. `lo` may be
/// above `hi` for a reversed axis (a horizontal bar chart reads top down).
#[derive(Clone, Copy, Debug)]
pub struct Scale {
    pub kind: Kind,
    pub lo: f64,
    pub hi: f64,
}

impl Scale {
    pub fn linear(lo: f64, hi: f64) -> R<Scale> {
        if !lo.is_finite() || !hi.is_finite() || lo == hi {
            return Err(format!("linear axis range [{lo}, {hi}] is empty or not finite"));
        }
        Ok(Scale {
            kind: Kind::Linear,
            lo,
            hi,
        })
    }

    pub fn log(lo: f64, hi: f64) -> R<Scale> {
        if !lo.is_finite() || !hi.is_finite() || lo <= 0.0 || hi <= 0.0 || lo == hi {
            return Err(format!(
                "log axis range [{lo}, {hi}] must be finite, above zero and not empty"
            ));
        }
        Ok(Scale {
            kind: Kind::Log,
            lo,
            hi,
        })
    }

    pub fn min(&self) -> f64 {
        self.lo.min(self.hi)
    }

    pub fn max(&self) -> f64 {
        self.lo.max(self.hi)
    }

    pub fn clamp(&self, v: f64) -> f64 {
        v.max(self.min()).min(self.max())
    }

    /// Whether `v` lies on the axis.
    pub fn contains(&self, v: f64) -> bool {
        let slack = (self.max() - self.min()) * 1e-9;
        v >= self.min() - slack && v <= self.max() + slack
    }

    /// Position of `v` along the axis, 0 at `lo` and 1 at `hi`.
    pub fn frac(&self, v: f64) -> f64 {
        match self.kind {
            Kind::Linear => (v - self.lo) / (self.hi - self.lo),
            Kind::Log => {
                let v = v.max(self.min());
                (v.ln() - self.lo.ln()) / (self.hi.ln() - self.lo.ln())
            }
        }
    }

    /// The range extended past its high end so that `px` pixels remain above
    /// `top` on an axis `len` pixels long: room for a label over the tallest
    /// mark, computed rather than guessed.
    pub fn headroom(kind: Kind, lo: f64, top: f64, len: f64, px: f64) -> R<Scale> {
        let share = px / (len - px).max(1.0);
        match kind {
            Kind::Linear => Scale::linear(lo, top + (top - lo) * share),
            Kind::Log => Scale::log(lo, top * (top / lo).powf(share)),
        }
    }
}

/// `v` as a tick label: the shortest spelling of its 11 leading digits, so
/// `0.30000000000000004` reads `0.3`, with `k` and `M` for thousands and
/// millions (`20000000` reads `20M`).
pub fn fmt_tick(v: f64) -> String {
    let plain = |x: f64| -> String {
        let rounded: f64 = format!("{x:.10e}").parse().unwrap_or(x);
        format!("{rounded}")
    };
    if v == 0.0 {
        "0".to_string()
    } else if v.abs() >= 1e6 {
        format!("{}M", plain(v / 1e6))
    } else if v.abs() >= 1e4 {
        format!("{}k", plain(v / 1e3))
    } else {
        plain(v)
    }
}

fn sorted_range(lo: f64, hi: f64) -> R<(f64, f64)> {
    if !lo.is_finite() || !hi.is_finite() {
        return Err(format!("axis range [{lo}, {hi}] is not finite"));
    }
    let (a, b) = (lo.min(hi), lo.max(hi));
    if a == b {
        return Err(format!("axis range [{lo}, {hi}] is empty"));
    }
    Ok((a, b))
}

/// Round-number ticks (steps of 1, 2 or 5 times a power of ten) covering
/// roughly `target` divisions of `[lo, hi]`. Never fewer than two.
pub fn linear_ticks(lo: f64, hi: f64, target: usize) -> R<Vec<f64>> {
    let (a, b) = sorted_range(lo, hi)?;
    let raw = (b - a) / target.max(2) as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let norm = raw / mag;
    let step = mag
        * if norm <= 1.0 {
            1.0
        } else if norm <= 2.0 {
            2.0
        } else if norm <= 5.0 {
            5.0
        } else {
            10.0
        };
    let k0 = (a / step - 1e-9).ceil() as i64;
    let k1 = (b / step + 1e-9).floor() as i64;
    let ticks: Vec<f64> = (k0..=k1).map(|k| k as f64 * step).collect();
    if ticks.len() < 2 {
        return Ok(vec![a, b]);
    }
    Ok(ticks)
}

/// Ticks for a log axis over `[lo, hi]`: the powers of ten when the range spans
/// several decades (every few decades past a dozen), 1-3, 1-2-5 or 1-2-3-5-7
/// within a decade or two, and round linear ticks when the range is narrower
/// than a decade and holds no such value. Always at least two, always inside
/// the range.
pub fn log_ticks(lo: f64, hi: f64) -> R<Vec<f64>> {
    let (a, b) = sorted_range(lo, hi)?;
    if a <= 0.0 {
        return Err(format!("log axis range [{lo}, {hi}] must be above zero"));
    }
    let e0 = a.log10().floor() as i32;
    let e1 = b.log10().ceil() as i32;
    let collect = |mults: &[f64], stride: i32| -> Vec<f64> {
        let mut out = Vec::new();
        let mut e = e0;
        while e <= e1 {
            for m in mults {
                let t = m * 10f64.powi(e);
                if t >= a * (1.0 - 1e-9) && t <= b * (1.0 + 1e-9) {
                    out.push(t);
                }
            }
            e += stride;
        }
        out
    };
    // Powers of ten when there are enough of them (thinned to every few
    // decades past ten); otherwise 1-2-5, 1-3 or 1-2-3-5-7, the coarsest that
    // gives four to twelve ticks.
    let mut t = collect(&[1.0], 1);
    if t.len() > 10 {
        let stride = (t.len() as f64 / 8.0).ceil() as i32;
        t = collect(&[1.0], stride);
    }
    if t.len() >= 4 {
        return Ok(t);
    }
    let mut finest = t;
    for mults in [
        &[1.0, 2.0, 5.0][..],
        &[1.0, 3.0],
        &[1.0, 2.0, 3.0, 5.0, 7.0],
        &[1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    ] {
        let t = collect(mults, 1);
        if (4..=12).contains(&t.len()) {
            return Ok(t);
        }
        if t.len() > finest.len() && t.len() <= 12 {
            finest = t;
        }
    }
    if finest.len() >= 3 {
        return Ok(finest);
    }
    linear_ticks(a, b, 5)
}

/// The largest 1, 2 or 5 times a power of ten that is at most `v`.
pub fn nice_floor(v: f64) -> R<f64> {
    if !(v.is_finite() && v > 0.0) {
        return Err(format!("{v} has no round-number floor"));
    }
    let mag = 10f64.powf(v.log10().floor());
    Ok([5.0, 2.0, 1.0]
        .iter()
        .map(|m| m * mag)
        .find(|t| *t <= v * (1.0 + 1e-12))
        .unwrap_or(mag))
}

/// The smallest 1, 2 or 5 times a power of ten that is at least `v`.
pub fn nice_ceil(v: f64) -> R<f64> {
    if !(v.is_finite() && v > 0.0) {
        return Err(format!("{v} has no round-number ceiling"));
    }
    let mag = 10f64.powf(v.log10().floor());
    Ok([1.0, 2.0, 5.0, 10.0]
        .iter()
        .map(|m| m * mag)
        .find(|t| *t >= v * (1.0 - 1e-12))
        .unwrap_or(10.0 * mag))
}

/// An upper bound for a linear axis from zero over data up to `max`: the next
/// whole tick step above `max`, so the top gridline sits just over the data.
pub fn nice_top(max: f64) -> R<f64> {
    if !(max.is_finite() && max > 0.0) {
        return Err(format!("{max} has no round-number top"));
    }
    let ticks = linear_ticks(0.0, max, 6)?;
    let step = ticks[1] - ticks[0];
    Ok((max / step - 1e-9).ceil() * step)
}

/// Offset of item `i` of `n` from the centre of its group, `step` apart.
pub fn spread(i: usize, n: usize, step: f64) -> f64 {
    (i as f64 - (n as f64 - 1.0) / 2.0) * step
}

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anchor {
    Start,
    Middle,
    End,
}

/// Where the `y` of a text call sits against its block of lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Va {
    /// The first line's baseline.
    Base,
    /// The last line's baseline (the block grows upward).
    Bottom,
    /// The top of the first line.
    Top,
    /// The vertical middle of the block.
    Middle,
}

#[derive(Clone, Copy, Debug)]
pub struct Ts {
    pub size: f64,
    pub anchor: Anchor,
    pub class: &'static str,
    pub bold: bool,
    pub rot: f64,
    pub va: Va,
}

impl Ts {
    pub fn new(size: f64) -> Ts {
        Ts {
            size,
            anchor: Anchor::Start,
            class: "mut",
            bold: false,
            rot: 0.0,
            va: Va::Base,
        }
    }

    pub fn mid(mut self) -> Ts {
        self.anchor = Anchor::Middle;
        self
    }

    pub fn end(mut self) -> Ts {
        self.anchor = Anchor::End;
        self
    }

    pub fn ink(mut self) -> Ts {
        self.class = "ink";
        self
    }

    pub fn dim(mut self) -> Ts {
        self.class = "dim";
        self
    }

    pub fn bold(mut self) -> Ts {
        self.bold = true;
        self
    }

    /// Counter-clockwise degrees, as in matplotlib.
    pub fn rot(mut self, deg: f64) -> Ts {
        self.rot = deg;
        self
    }

    pub fn va(mut self, va: Va) -> Ts {
        self.va = va;
        self
    }
}

fn line_height(size: f64) -> f64 {
    size * 1.2
}

/// One `<text>` element; `s` may hold `\n` breaks, set as stacked lines.
fn emit_text(buf: &mut String, x: f64, y: f64, s: &str, ts: &Ts) {
    let lines: Vec<&str> = s.split('\n').collect();
    let lh = line_height(ts.size);
    let n = lines.len() as f64;
    let y0 = if ts.rot != 0.0 {
        y
    } else {
        match ts.va {
            Va::Base => y,
            Va::Bottom => y - (n - 1.0) * lh,
            Va::Top => y + 0.8 * ts.size,
            Va::Middle => y + 0.35 * ts.size - (n - 1.0) * lh / 2.0,
        }
    };
    let anchor = match ts.anchor {
        Anchor::Start => "",
        Anchor::Middle => " text-anchor=\"middle\"",
        Anchor::End => " text-anchor=\"end\"",
    };
    let weight = if ts.bold { " font-weight=\"600\"" } else { "" };
    let rotate = if ts.rot != 0.0 {
        format!(" transform=\"rotate({} {} {})\"", num(-ts.rot), num(x), num(y0))
    } else {
        String::new()
    };
    buf.push_str(&format!(
        "<text class=\"{}\" x=\"{}\" y=\"{}\" font-size=\"{}\"{anchor}{weight}{rotate}>",
        ts.class,
        num(x),
        num(y0),
        num(ts.size)
    ));
    if lines.len() == 1 {
        buf.push_str(&esc(lines[0]));
    } else {
        for (i, l) in lines.iter().enumerate() {
            buf.push_str(&format!(
                "<tspan x=\"{}\" dy=\"{}\">{}</tspan>",
                num(x),
                if i == 0 { "0".to_string() } else { num(lh) },
                esc(l)
            ));
        }
    }
    buf.push_str("</text>\n");
}

// ---------------------------------------------------------------------------
// Paint
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Edge {
    None,
    Bg(f64),
}

/// How a filled mark looks.
#[derive(Clone, Debug)]
pub struct Paint {
    fill: String,
    alpha: f64,
    edge: Edge,
}

impl Paint {
    pub fn new(fill: &str) -> Paint {
        Paint {
            fill: fill.to_string(),
            alpha: 1.0,
            edge: Edge::None,
        }
    }

    pub fn alpha(mut self, a: f64) -> Paint {
        self.alpha = a;
        self
    }

    /// An outline in the background colour, to separate neighbours.
    pub fn bg_edge(mut self, w: f64) -> Paint {
        self.edge = Edge::Bg(w);
        self
    }

    /// The ink colour (light on a dark page).
    pub fn ink() -> Paint {
        Paint::new("")
    }

    /// The `class`, `fill` and outline attributes of a mark of kind `class`.
    fn attrs(&self, class: &str) -> String {
        let ink = if self.fill.is_empty() { " ink" } else { "" };
        let mut s = match self.edge {
            Edge::Bg(_) => format!(" class=\"{class}{ink} edge\""),
            _ => format!(" class=\"{class}{ink}\""),
        };
        if !self.fill.is_empty() {
            s.push_str(&format!(" fill=\"{}\"", esc(&self.fill)));
        }
        if self.alpha < 1.0 {
            s.push_str(&format!(" fill-opacity=\"{}\"", num(self.alpha)));
        }
        match &self.edge {
            Edge::None => {}
            Edge::Bg(w) => s.push_str(&format!(" stroke-width=\"{}\"", num(*w))),
        }
        s
    }
}

/// How a line looks.
#[derive(Clone, Debug)]
pub struct Stroke {
    color: Option<String>,
    muted: bool,
    w: f64,
    dash: bool,
    alpha: f64,
}

impl Stroke {
    /// The ink colour (light on a dark page).
    pub fn ink(w: f64) -> Stroke {
        Stroke {
            color: None,
            muted: false,
            w,
            dash: false,
            alpha: 1.0,
        }
    }

    pub fn muted(w: f64) -> Stroke {
        Stroke {
            color: None,
            muted: true,
            w,
            dash: false,
            alpha: 1.0,
        }
    }

    pub fn color(c: &str, w: f64) -> Stroke {
        Stroke {
            color: Some(c.to_string()),
            muted: false,
            w,
            dash: false,
            alpha: 1.0,
        }
    }

    pub fn dashed(mut self) -> Stroke {
        self.dash = true;
        self
    }

    pub fn alpha(mut self, a: f64) -> Stroke {
        self.alpha = a;
        self
    }

    /// The `class` and stroke attributes of a line of kind `class`.
    fn attrs(&self, class: &str) -> String {
        let mut s = match (&self.color, self.muted) {
            (Some(c), _) => format!(" class=\"{class}\" stroke=\"{}\"", esc(c)),
            (None, true) => format!(" class=\"{class} mk\""),
            (None, false) => format!(" class=\"{class} sk\""),
        };
        s.push_str(&format!(" stroke-width=\"{}\"", num(self.w)));
        if self.dash {
            s.push_str(" stroke-dasharray=\"5 4\"");
        }
        if self.alpha < 1.0 {
            s.push_str(&format!(" stroke-opacity=\"{}\"", num(self.alpha)));
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Legends
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Swatch {
    Box { fill: String, alpha: f64 },
    Dot { fill: String },
    Diamond { fill: String },
    Cross { color: String },
    Line { color: String, dash: bool },
}

#[derive(Clone, Debug)]
pub struct LegendItem {
    pub label: String,
    pub swatch: Swatch,
}

impl LegendItem {
    pub fn boxed(label: &str, fill: &str) -> LegendItem {
        LegendItem {
            label: label.to_string(),
            swatch: Swatch::Box {
                fill: fill.to_string(),
                alpha: 1.0,
            },
        }
    }

    pub fn faint(label: &str, fill: &str, alpha: f64) -> LegendItem {
        LegendItem {
            label: label.to_string(),
            swatch: Swatch::Box {
                fill: fill.to_string(),
                alpha,
            },
        }
    }

    pub fn dot(label: &str, fill: &str) -> LegendItem {
        LegendItem {
            label: label.to_string(),
            swatch: Swatch::Dot { fill: fill.to_string() },
        }
    }

    pub fn diamond(label: &str, fill: &str) -> LegendItem {
        LegendItem {
            label: label.to_string(),
            swatch: Swatch::Diamond { fill: fill.to_string() },
        }
    }

    pub fn cross(label: &str, color: &str) -> LegendItem {
        LegendItem {
            label: label.to_string(),
            swatch: Swatch::Cross {
                color: color.to_string(),
            },
        }
    }

    pub fn line(label: &str, color: &str, dash: bool) -> LegendItem {
        LegendItem {
            label: label.to_string(),
            swatch: Swatch::Line {
                color: color.to_string(),
                dash,
            },
        }
    }

    /// A line in the ink colour, for a mark drawn with [`Stroke::ink`]: it
    /// follows the theme as that mark does (a literal near-black would vanish
    /// on the dark background).
    pub fn ink_line(label: &str, dash: bool) -> LegendItem {
        LegendItem::line(label, "", dash)
    }
}

const LEGEND_ROW: f64 = 17.0;

/// Rows of item indexes and their x offsets, filled left to right and wrapped
/// at `max_w`.
fn legend_rows(items: &[LegendItem], max_w: f64) -> Vec<Vec<(usize, f64)>> {
    let mut rows: Vec<Vec<(usize, f64)>> = Vec::new();
    let mut x = 0.0;
    for (i, it) in items.iter().enumerate() {
        let w = 16.0 + 6.0 + text_width(&it.label, TICK) + 16.0;
        if rows.is_empty() || (x + w > max_w && x > 0.0) {
            rows.push(Vec::new());
            x = 0.0;
        }
        if let Some(row) = rows.last_mut() {
            row.push((i, x));
        }
        x += w;
    }
    rows
}

fn draw_legend(buf: &mut String, x: f64, y: f64, items: &[LegendItem], rows: &[Vec<(usize, f64)>]) {
    for (r, row) in rows.iter().enumerate() {
        let cy = y + r as f64 * LEGEND_ROW + LEGEND_ROW / 2.0 - 1.0;
        for (i, dx) in row {
            let it = &items[*i];
            let sx = x + dx;
            match &it.swatch {
                Swatch::Box { fill, alpha } => buf.push_str(&format!(
                    "<rect class=\"key\" x=\"{}\" y=\"{}\" width=\"14\" height=\"10\" fill=\"{}\"{}/>\n",
                    num(sx),
                    num(cy - 5.0),
                    esc(fill),
                    if *alpha < 1.0 {
                        format!(" fill-opacity=\"{}\"", num(*alpha))
                    } else {
                        String::new()
                    }
                )),
                Swatch::Dot { fill } => buf.push_str(&format!(
                    "<circle class=\"key\" cx=\"{}\" cy=\"{}\" r=\"4\" fill=\"{}\"/>\n",
                    num(sx + 7.0),
                    num(cy),
                    esc(fill)
                )),
                Swatch::Diamond { fill } => buf.push_str(&format!(
                    "<path class=\"key\" d=\"{}\" fill=\"{}\"/>\n",
                    diamond_path(sx + 7.0, cy, 5.0),
                    esc(fill)
                )),
                Swatch::Cross { color } => buf.push_str(&format!(
                    "<path class=\"key\" d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.4\"/>\n",
                    cross_path(sx + 7.0, cy, 3.5),
                    esc(color)
                )),
                Swatch::Line { color, dash } => buf.push_str(&format!(
                    "<line {} x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke-width=\"1.6\"{}/>\n",
                    if color.is_empty() {
                        "class=\"key sk\"".to_string()
                    } else {
                        format!("class=\"key\" stroke=\"{}\"", esc(color))
                    },
                    num(sx),
                    num(cy),
                    num(sx + 14.0),
                    num(cy),
                    if *dash { " stroke-dasharray=\"4 3\"" } else { "" }
                )),
            }
            emit_text(buf, sx + 20.0, cy + 4.0, &it.label, &Ts::new(TICK));
        }
    }
}

fn diamond_path(cx: f64, cy: f64, r: f64) -> String {
    format!(
        "M{} {}L{} {}L{} {}L{} {}Z",
        num(cx),
        num(cy - r),
        num(cx + r),
        num(cy),
        num(cx),
        num(cy + r),
        num(cx - r),
        num(cy)
    )
}

fn cross_path(cx: f64, cy: f64, r: f64) -> String {
    format!(
        "M{} {}L{} {}M{} {}L{} {}",
        num(cx - r),
        num(cy - r),
        num(cx + r),
        num(cy + r),
        num(cx - r),
        num(cy + r),
        num(cx + r),
        num(cy - r)
    )
}

// ---------------------------------------------------------------------------
// Axes
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[derive(Clone, Debug)]
pub struct Tick {
    pub v: f64,
    pub label: String,
}

/// One axis: its scale, tick positions and labels, whether it draws grid
/// lines, its title and the slant of its tick labels.
#[derive(Clone, Debug)]
pub struct AxisCfg {
    pub scale: Scale,
    pub ticks: Vec<Tick>,
    pub grid: bool,
    pub label: String,
    pub rot: f64,
    categorical: bool,
}

fn numeric_ticks(scale: &Scale) -> R<Vec<Tick>> {
    let vals = match scale.kind {
        Kind::Linear => linear_ticks(scale.lo, scale.hi, 6)?,
        Kind::Log => log_ticks(scale.lo, scale.hi)?,
    };
    Ok(vals.into_iter().map(|v| Tick { v, label: fmt_tick(v) }).collect())
}

impl AxisCfg {
    /// A numeric axis over an existing scale, with round ticks and grid lines.
    pub fn from_scale(scale: Scale) -> R<AxisCfg> {
        Ok(AxisCfg {
            ticks: numeric_ticks(&scale)?,
            scale,
            grid: true,
            label: String::new(),
            rot: 0.0,
            categorical: false,
        })
    }

    /// A linear numeric axis with round ticks and grid lines.
    pub fn linear(lo: f64, hi: f64) -> R<AxisCfg> {
        let scale = Scale::linear(lo, hi)?;
        Ok(AxisCfg {
            ticks: numeric_ticks(&scale)?,
            scale,
            grid: true,
            label: String::new(),
            rot: 0.0,
            categorical: false,
        })
    }

    /// A log numeric axis with round ticks and grid lines.
    pub fn log(lo: f64, hi: f64) -> R<AxisCfg> {
        let scale = Scale::log(lo, hi)?;
        Ok(AxisCfg {
            ticks: numeric_ticks(&scale)?,
            scale,
            grid: true,
            label: String::new(),
            rot: 0.0,
            categorical: false,
        })
    }

    /// A numeric axis with exactly the ticks asked for (those inside the range).
    pub fn fixed(scale: Scale, ticks: &[f64]) -> AxisCfg {
        AxisCfg {
            ticks: ticks
                .iter()
                .filter(|v| scale.contains(**v))
                .map(|v| Tick {
                    v: *v,
                    label: fmt_tick(*v),
                })
                .collect(),
            scale,
            grid: true,
            label: String::new(),
            rot: 0.0,
            categorical: false,
        }
    }

    /// Categories at positions 0..n. `reverse` puts the first at the top of a
    /// vertical axis, which is how a horizontal bar chart is read.
    pub fn categories(labels: &[String], reverse: bool) -> R<AxisCfg> {
        if labels.is_empty() {
            return Err("an axis of zero categories".to_string());
        }
        let n = labels.len() as f64;
        let scale = if reverse {
            Scale::linear(n - 0.5, -0.5)?
        } else {
            Scale::linear(-0.5, n - 0.5)?
        };
        Ok(AxisCfg {
            scale,
            ticks: labels
                .iter()
                .enumerate()
                .map(|(i, l)| Tick {
                    v: i as f64,
                    label: l.clone(),
                })
                .collect(),
            grid: false,
            label: String::new(),
            rot: 0.0,
            categorical: true,
        })
    }

    pub fn labelled(mut self, label: &str) -> AxisCfg {
        self.label = label.to_string();
        self
    }

    /// Slant the tick labels by `deg` degrees (reading upward to the right).
    pub fn rotated(mut self, deg: f64) -> AxisCfg {
        self.rot = deg;
        self
    }
}

/// A panel: a plot area with its frame already drawn, ready for marks.
pub struct Axes {
    plot: Rect,
    x: Scale,
    y: Scale,
    buf: String,
}

const MARGIN_R: f64 = 12.0;

impl Axes {
    /// Lay out the panel inside `cell` (titles, legend, axis labels and tick
    /// labels take what they need; the plot area is what remains) and draw the
    /// grid, axes, ticks and text. Marks are added afterwards.
    pub fn new(cell: Rect, title: &str, legend: &[LegendItem], xc: AxisCfg, yc: AxisCfg) -> R<Axes> {
        let title_lines = if title.is_empty() {
            Vec::new()
        } else {
            wrap(title, cell.w - 10.0, TITLE)
        };
        let title_h = title_lines.len() as f64 * line_height(TITLE) + if title_lines.is_empty() { 0.0 } else { 4.0 };
        let rows = legend_rows(legend, cell.w - 10.0);
        let legend_h = rows.len() as f64 * LEGEND_ROW;
        let top = 6.0 + title_h + legend_h + if title_h + legend_h > 0.0 { 8.0 } else { 0.0 };

        // Left: y axis title (rotated), tick labels, a tick mark.
        let ylabel_lines = if yc.label.is_empty() {
            Vec::new()
        } else {
            wrap(&yc.label, cell.h * 0.8, LABEL)
        };
        let y_tick_w = yc
            .ticks
            .iter()
            .flat_map(|t| t.label.split('\n').map(|l| text_width(l, TICK)))
            .fold(0.0, f64::max);
        let left0 = 6.0 + ylabel_lines.len() as f64 * line_height(LABEL) + y_tick_w + 12.0;

        // Bottom: x tick labels, as slanted or as wide as a category's slot.
        let n_x = xc.ticks.len().max(1) as f64;
        let mut left = left0;
        let mut x_lines: Vec<Vec<String>> = Vec::new();
        let mut x_rot = xc.rot;
        let mut x_label_h = 0.0;
        for _ in 0..3 {
            let plot_w = (cell.w - left - MARGIN_R).max(1.0);
            let slot = plot_w / n_x;
            let widest = xc
                .ticks
                .iter()
                .flat_map(|t| t.label.split('\n').map(|l| text_width(l, TICK)))
                .fold(0.0, f64::max);
            x_rot = xc.rot;
            if x_rot == 0.0 && xc.categorical && widest > slot * 0.96 {
                let any_space = xc.ticks.iter().any(|t| t.label.contains(' '));
                if !any_space
                    || xc
                        .ticks
                        .iter()
                        .any(|t| t.label.split(' ').any(|w| text_width(w, TICK) > slot * 0.96))
                {
                    x_rot = 30.0;
                }
            }
            x_lines = xc
                .ticks
                .iter()
                .map(|t| {
                    if x_rot == 0.0 && xc.categorical {
                        wrap(&t.label, slot * 0.96, TICK)
                    } else {
                        t.label.split('\n').map(str::to_string).collect()
                    }
                })
                .collect();
            let (sin, cos) = (x_rot.to_radians().sin(), x_rot.to_radians().cos());
            x_label_h = x_lines
                .iter()
                .zip(&xc.ticks)
                .map(|(ls, _)| {
                    let w = ls.iter().map(|l| text_width(l, TICK)).fold(0.0, f64::max);
                    let h = ls.len() as f64 * line_height(TICK);
                    if x_rot == 0.0 {
                        h
                    } else {
                        w * sin + h * cos
                    }
                })
                .fold(0.0, f64::max);
            // A slanted first label hangs left of its tick.
            if x_rot != 0.0 {
                let first_w = x_lines
                    .first()
                    .map(|ls| ls.iter().map(|l| text_width(l, TICK)).fold(0.0, f64::max))
                    .unwrap_or(0.0);
                let tick_x = xc.ticks.first().map(|t| xc.scale.frac(t.v)).unwrap_or(0.0) * plot_w;
                let need = first_w * x_rot.to_radians().cos() + 8.0 - tick_x;
                left = left0.max(left0 + need);
            } else {
                left = left0;
            }
        }
        let xlabel_lines = if xc.label.is_empty() {
            Vec::new()
        } else {
            wrap(&xc.label, cell.w * 0.9, LABEL)
        };
        let bottom = 8.0
            + x_label_h
            + 6.0
            + xlabel_lines.len() as f64 * line_height(LABEL)
            + if xlabel_lines.is_empty() { 0.0 } else { 4.0 };

        let plot = Rect {
            x: cell.x + left,
            y: cell.y + top,
            w: cell.w - left - MARGIN_R,
            h: cell.h - top - bottom,
        };
        if plot.w < 60.0 || plot.h < 60.0 {
            return Err(format!(
                "panel {:?} is too small for its labels: {:.0}x{:.0} px left for the plot",
                title, plot.w, plot.h
            ));
        }
        let mut ax = Axes {
            plot,
            x: xc.scale,
            y: yc.scale,
            buf: String::new(),
        };
        ax.buf.push_str("<g class=\"panel\">\n");

        // Grid and ticks.
        for t in &yc.ticks {
            let py = ax.py(t.v);
            if yc.grid {
                ax.buf.push_str(&format!(
                    "<line class=\"grid\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"/>\n",
                    num(plot.x),
                    num(py),
                    num(plot.x + plot.w),
                    num(py)
                ));
            }
            ax.buf.push_str(&format!(
                "<line class=\"axis\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"/>\n",
                num(plot.x - 4.0),
                num(py),
                num(plot.x),
                num(py)
            ));
            emit_text(
                &mut ax.buf,
                plot.x - 7.0,
                py,
                &t.label,
                &Ts::new(TICK).end().va(Va::Middle),
            );
        }
        for (t, lines) in xc.ticks.iter().zip(&x_lines) {
            let px = ax.px(t.v);
            if xc.grid {
                ax.buf.push_str(&format!(
                    "<line class=\"grid\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"/>\n",
                    num(px),
                    num(plot.y),
                    num(px),
                    num(plot.y + plot.h)
                ));
            }
            ax.buf.push_str(&format!(
                "<line class=\"axis\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"/>\n",
                num(px),
                num(plot.y + plot.h),
                num(px),
                num(plot.y + plot.h + 4.0)
            ));
            let text = lines.join("\n");
            if x_rot == 0.0 {
                emit_text(
                    &mut ax.buf,
                    px,
                    plot.y + plot.h + 8.0,
                    &text,
                    &Ts::new(TICK).mid().va(Va::Top),
                );
            } else {
                // Slanted labels end at the tick, as matplotlib's `ha="right"`.
                emit_text(
                    &mut ax.buf,
                    px + 3.0,
                    plot.y + plot.h + 14.0,
                    &text.replace('\n', " "),
                    &Ts::new(TICK).end().rot(x_rot),
                );
            }
        }
        ax.buf.push_str(&format!(
            "<line class=\"axis\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"/>\n",
            num(plot.x),
            num(plot.y + plot.h),
            num(plot.x + plot.w),
            num(plot.y + plot.h)
        ));
        ax.buf.push_str(&format!(
            "<line class=\"axis\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"/>\n",
            num(plot.x),
            num(plot.y),
            num(plot.x),
            num(plot.y + plot.h)
        ));

        // Titles.
        for (i, l) in ylabel_lines.iter().enumerate() {
            let cx = cell.x + 6.0 + (i as f64 + 0.8) * line_height(LABEL);
            emit_text(
                &mut ax.buf,
                cx,
                plot.y + plot.h / 2.0,
                l,
                &Ts::new(LABEL).mid().rot(90.0),
            );
        }
        for (i, l) in xlabel_lines.iter().enumerate() {
            emit_text(
                &mut ax.buf,
                plot.x + plot.w / 2.0,
                cell.y + cell.h - 6.0 - (xlabel_lines.len() - 1 - i) as f64 * line_height(LABEL),
                l,
                &Ts::new(LABEL).mid(),
            );
        }
        for (i, l) in title_lines.iter().enumerate() {
            emit_text(
                &mut ax.buf,
                plot.x,
                cell.y + 6.0 + (i as f64 + 0.85) * line_height(TITLE),
                l,
                &Ts::new(TITLE).ink().bold(),
            );
        }
        draw_legend(&mut ax.buf, plot.x, cell.y + 6.0 + title_h, legend, &rows);
        Ok(ax)
    }

    pub fn px(&self, x: f64) -> f64 {
        self.plot.x + self.x.frac(x) * self.plot.w
    }

    pub fn py(&self, y: f64) -> f64 {
        self.plot.y + self.plot.h - self.y.frac(y) * self.plot.h
    }

    fn inside(&self, x: f64, y: f64) -> bool {
        self.x.contains(x) && self.y.contains(y)
    }

    /// A rectangle spanning `[x0, x1] x [y0, y1]` in data coordinates, clipped
    /// to the axes (on a log axis a bar from "zero" starts at the bottom).
    pub fn rect(&mut self, class: &str, xs: (f64, f64), ys: (f64, f64), paint: &Paint) {
        let (xa, xb) = (self.x.clamp(xs.0), self.x.clamp(xs.1));
        let (ya, yb) = (self.y.clamp(ys.0), self.y.clamp(ys.1));
        let (px0, px1) = (self.px(xa), self.px(xb));
        let (py0, py1) = (self.py(ya), self.py(yb));
        let w = (px1 - px0).abs();
        let h = (py1 - py0).abs();
        if w == 0.0 || h == 0.0 {
            return;
        }
        self.buf.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"{}/>\n",
            num(px0.min(px1)),
            num(py0.min(py1)),
            num(w),
            num(h),
            paint.attrs(class)
        ));
    }

    /// A vertical bar of data width `w` centred on `x`, from `y0` to `y1`.
    pub fn bar(&mut self, x: f64, w: f64, ys: (f64, f64), paint: &Paint) {
        self.rect("bar", (x - w / 2.0, x + w / 2.0), ys, paint);
    }

    /// A horizontal bar of data height `h` centred on `y`, from `x0` to `x1`.
    pub fn hbar(&mut self, y: f64, h: f64, xs: (f64, f64), paint: &Paint) {
        self.rect("bar", xs, (y - h / 2.0, y + h / 2.0), paint);
    }

    /// One segment of a stacked bar.
    pub fn seg(&mut self, x: f64, w: f64, ys: (f64, f64), paint: &Paint) {
        self.rect("seg", (x - w / 2.0, x + w / 2.0), ys, paint);
    }

    /// One segment of a stacked horizontal bar.
    pub fn hseg(&mut self, y: f64, h: f64, xs: (f64, f64), paint: &Paint) {
        self.rect("seg", xs, (y - h / 2.0, y + h / 2.0), paint);
    }

    /// A segment between two data points.
    pub fn line(&mut self, class: &str, from: (f64, f64), to: (f64, f64), stroke: &Stroke) {
        if !self.inside(from.0, from.1) || !self.inside(to.0, to.1) {
            return;
        }
        self.buf.push_str(&format!(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"{}/>\n",
            num(self.px(from.0)),
            num(self.py(from.1)),
            num(self.px(to.0)),
            num(self.py(to.1)),
            stroke.attrs(class)
        ));
    }

    /// A reference line across the whole plot at height `y`.
    pub fn axhline(&mut self, y: f64, stroke: &Stroke) {
        if !self.y.contains(y) {
            return;
        }
        let py = self.py(y);
        self.buf.push_str(&format!(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"{}/>\n",
            num(self.plot.x),
            num(py),
            num(self.plot.x + self.plot.w),
            num(py),
            stroke.attrs("ref")
        ));
    }

    /// A circle of radius `r` px at a data point, skipped when off the axes.
    pub fn dot(&mut self, x: f64, y: f64, r: f64, paint: &Paint) {
        self.dot_px(x, y, 0.0, r, paint);
    }

    /// A dot shifted sideways by `dx` px from its data point.
    pub fn dot_px(&mut self, x: f64, y: f64, dx: f64, r: f64, paint: &Paint) {
        if !self.inside(x, y) {
            return;
        }
        self.buf.push_str(&format!(
            "<circle cx=\"{}\" cy=\"{}\" r=\"{}\"{}/>\n",
            num(self.px(x) + dx),
            num(self.py(y)),
            num(r),
            paint.attrs("dot")
        ));
    }

    pub fn diamond(&mut self, x: f64, y: f64, r: f64, paint: &Paint) {
        if !self.inside(x, y) {
            return;
        }
        self.buf.push_str(&format!(
            "<path d=\"{}\"{}/>\n",
            diamond_path(self.px(x), self.py(y), r),
            paint.attrs("marker")
        ));
    }

    /// An `x` marker of half-size `r` px.
    pub fn cross(&mut self, x: f64, y: f64, r: f64, color: &str) {
        if !self.inside(x, y) {
            return;
        }
        self.buf.push_str(&format!(
            "<path class=\"marker\" d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.4\"/>\n",
            cross_path(self.px(x), self.py(y), r),
            esc(color)
        ));
    }

    /// A short horizontal tick of half-width `half` px centred on the point.
    pub fn tick(&mut self, x: f64, y: f64, half: f64, stroke: &Stroke) {
        if !self.inside(x, y) {
            return;
        }
        let (px, py) = (self.px(x), self.py(y));
        self.buf.push_str(&format!(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"{}/>\n",
            num(px - half),
            num(py),
            num(px + half),
            num(py),
            stroke.attrs("marker")
        ));
    }

    /// A vertical whisker from `lo` to `hi` at `x`, capped with `cap` px bars.
    pub fn whisker(&mut self, x: f64, ys: (f64, f64), cap: f64, stroke: &Stroke) {
        let (lo, hi) = (self.y.clamp(ys.0.min(ys.1)), self.y.clamp(ys.0.max(ys.1)));
        if !self.x.contains(x) {
            return;
        }
        let (px, p0, p1) = (self.px(x), self.py(lo), self.py(hi));
        let a = stroke.attrs("whisker");
        self.buf.push_str(&format!(
            "<g{a}><line x1=\"{px}\" y1=\"{p0}\" x2=\"{px}\" y2=\"{p1}\"/><line x1=\"{}\" y1=\"{p0}\" x2=\"{}\" y2=\"{p0}\"/><line x1=\"{}\" y1=\"{p1}\" x2=\"{}\" y2=\"{p1}\"/></g>\n",
            num(px - cap),
            num(px + cap),
            num(px - cap),
            num(px + cap),
            px = num(px),
            p0 = num(p0),
            p1 = num(p1),
        ));
    }

    /// Text at a data point, moved by a pixel offset; skipped when off the axes.
    pub fn text(&mut self, x: f64, y: f64, off: (f64, f64), s: &str, ts: &Ts) {
        if !self.inside(x, y) {
            return;
        }
        let (px, py) = (self.px(x) + off.0, self.py(y) + off.1);
        emit_text(&mut self.buf, px, py, s, ts);
    }

    /// Text at a fraction of the plot area (0, 0 bottom left), for notes that
    /// belong to the panel and not to a data point.
    pub fn note(&mut self, fx: f64, fy: f64, s: &str, ts: &Ts) {
        emit_text(
            &mut self.buf,
            self.plot.x + fx * self.plot.w,
            self.plot.y + (1.0 - fy) * self.plot.h,
            s,
            ts,
        );
    }

    fn finish(mut self) -> String {
        self.buf.push_str("</g>\n");
        self.buf
    }
}

// ---------------------------------------------------------------------------
// Figures
// ---------------------------------------------------------------------------

/// A whole figure: a headline and notes, an optional legend for all panels,
/// and a grid of panels below.
pub struct Figure {
    width: f64,
    header: f64,
    content_h: f64,
    head: String,
    body: String,
    title: String,
}

const PAD: f64 = 14.0;
const GAP_X: f64 = 22.0;
const GAP_Y: f64 = 26.0;

impl Figure {
    /// `content_h` is the height of the panel grid; the header (headline,
    /// notes, legend) is measured from the text and added above it.
    pub fn new(width: f64, content_h: f64, headline: &str, notes: &[&str], legend: &[LegendItem]) -> Figure {
        let mut head = String::new();
        let text_w = width - 2.0 * PAD;
        let mut y = 12.0;
        let hl = wrap(headline, text_w, HEADLINE);
        for l in &hl {
            y += line_height(HEADLINE);
            emit_text(&mut head, PAD, y - 4.0, l, &Ts::new(HEADLINE).ink().bold());
        }
        for note in notes {
            for l in wrap(note, text_w, TICK + 0.5) {
                y += line_height(TICK + 0.5);
                emit_text(&mut head, PAD, y - 3.0, &l, &Ts::new(TICK + 0.5));
            }
        }
        let rows = legend_rows(legend, text_w);
        if !rows.is_empty() {
            y += 8.0;
            draw_legend(&mut head, PAD, y, legend, &rows);
            y += rows.len() as f64 * LEGEND_ROW;
        }
        let header = y + 10.0;
        Figure {
            width,
            header,
            content_h,
            head,
            body: String::new(),
            title: headline.to_string(),
        }
    }

    /// Cells of a grid with the given column and row weights, row by row.
    pub fn cells(&self, col_w: &[f64], row_h: &[f64]) -> Vec<Rect> {
        let cw: f64 = col_w.iter().sum();
        let rh: f64 = row_h.iter().sum();
        let avail_w = self.width - 2.0 * PAD - GAP_X * (col_w.len() as f64 - 1.0);
        let avail_h = self.content_h - GAP_Y * (row_h.len() as f64 - 1.0);
        let mut out = Vec::new();
        let mut y = self.header;
        for h in row_h {
            let ch = avail_h * h / rh;
            let mut x = PAD;
            for w in col_w {
                let cwid = avail_w * w / cw;
                out.push(Rect { x, y, w: cwid, h: ch });
                x += cwid + GAP_X;
            }
            y += ch + GAP_Y;
        }
        out
    }

    pub fn add(&mut self, ax: Axes) {
        self.body.push_str(&ax.finish());
    }

    /// The finished SVG document.
    pub fn finish(self) -> String {
        let h = self.header + self.content_h + PAD;
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" role=\"img\">\n<title>{title}</title>\n<style>\n{STYLE}\n</style>\n<rect class=\"bg\" width=\"100%\" height=\"100%\"/>\n{head}{body}</svg>\n",
            w = num(self.width),
            h = num(h),
            title = esc(&self.title),
            head = self.head,
            body = self.body,
        )
    }
}
