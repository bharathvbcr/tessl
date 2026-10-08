//! Flash-attention shape sweep: the three attention kernels, JSON out for the
//! cross-runtime compare in `bench/flash_attn_torch_mlx.py`.
//!
//! These kernels had thorough correctness coverage (`tests/attention.rs` scores
//! them against an f64 transcription of their own masking rule) and **no timing
//! lane at all** — for the kernel that dominates inference cost.
//!
//! Protocol is pinned to `bench_gemm_sweep`: same env contract, synchronize
//! every iteration, median over iters. `--dump-parity DIR` writes Q/K/V and
//! every lane's O for the numeric check.
//!
//! FLOP accounting is *exact*, not nominal: the kernels skip masked work, so
//! the useful-work count is over unmasked (q, k) pairs only, computed from the
//! same rule the kernels apply. `dense_gflops` is reported alongside so a
//! reader can see how much of the square was skipped.

mod common;

use common::{env_usize, fill_dist, median, Dist};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tessl::nn::{self, AttnDims, AttnHeadDim};
use tessl::npy::write_npy_f32;
use tessl::runtime::GpuRuntime;
use tessl::tensor::GpuBuffer;

/// One attention configuration.
#[derive(Clone, Copy)]
struct Cfg {
    label: &'static str,
    b: usize,
    tq: usize,
    tkv: usize,
    h: usize,
    hkv: usize,
    d: usize,
    /// `None` selects the global (causal, D=512) kernel.
    window: Option<usize>,
    q_off: usize,
    kv_off: usize,
}

impl Cfg {
    fn scale(&self) -> f32 {
        1.0 / (self.d as f32).sqrt()
    }

    /// Unmasked (query, key) pairs, by the kernels' own rule:
    /// `q_abs = q_off + t_q`, `k_abs = kv_off + t_k`, sliding window keeps
    /// `max(0, q_abs - w + 1) <= k_abs <= q_abs`, global keeps `k_abs <= q_abs`.
    fn live_pairs(&self) -> u64 {
        let mut n = 0u64;
        for t_q in 0..self.tq {
            let q_abs = (self.q_off + t_q) as i64;
            let lo = match self.window {
                Some(w) => (q_abs - w as i64 + 1).max(0),
                None => 0,
            };
            for t_k in 0..self.tkv {
                let k_abs = (self.kv_off + t_k) as i64;
                if k_abs >= lo && k_abs <= q_abs {
                    n += 1;
                }
            }
        }
        n
    }

    /// 2*D for the QK dot and 2*D for the PV accumulate, per live pair, per
    /// batch and query head.
    fn live_flop(&self) -> f64 {
        4.0 * self.d as f64 * (self.b * self.h) as f64 * self.live_pairs() as f64
    }

    fn dense_flop(&self) -> f64 {
        4.0 * self.d as f64 * (self.b * self.h) as f64 * (self.tq * self.tkv) as f64
    }

    fn head_dim(&self) -> Result<AttnHeadDim, String> {
        match self.d {
            128 => Ok(AttnHeadDim::D128),
            256 => Ok(AttnHeadDim::D256),
            other => Err(format!(
                "{}: head dim {other} has no sliding-window kernel (128 or 256)",
                self.label
            )),
        }
    }

    fn dims(&self) -> AttnDims {
        AttnDims {
            batch: self.b as u32,
            tq: self.tq as u32,
            heads: self.h as u32,
            heads_kv: self.hkv as u32,
            window: self.window.unwrap_or(0) as u32,
            scale: self.scale(),
        }
    }

    fn q_elems(&self) -> usize {
        self.b * self.tq * self.h * self.d
    }

    fn kv_elems(&self) -> usize {
        self.b * self.tkv * self.hkv * self.d
    }
}

/// Prefill and decode at each head dimension. `window` of `None` routes to
/// `flash_attn_global_h512`, which is the only D=512 path.
///
/// Head counts follow the grouped-query ratios production models actually use
/// (H:Hkv of 4:1), because the kernel indexes K/V by `h / (H/Hkv)` and a 1:1
/// ratio would never exercise that divide.
const CFGS: &[Cfg] = &[
    // ---- D=128 sliding window, 4:1 GQA ----
    Cfg {
        label: "swa128_prefill_512",
        b: 1,
        tq: 512,
        tkv: 512,
        h: 32,
        hkv: 8,
        d: 128,
        window: Some(1024),
        q_off: 0,
        kv_off: 0,
    },
    Cfg {
        label: "swa128_prefill_2048",
        b: 1,
        tq: 2048,
        tkv: 2048,
        h: 32,
        hkv: 8,
        d: 128,
        window: Some(1024),
        q_off: 0,
        kv_off: 0,
    },
    Cfg {
        label: "swa128_prefill_4096",
        b: 1,
        tq: 4096,
        tkv: 4096,
        h: 32,
        hkv: 8,
        d: 128,
        window: Some(1024),
        q_off: 0,
        kv_off: 0,
    },
    Cfg {
        label: "swa128_decode_1k",
        b: 1,
        tq: 1,
        tkv: 1024,
        h: 32,
        hkv: 8,
        d: 128,
        window: Some(1024),
        q_off: 1023,
        kv_off: 0,
    },
    Cfg {
        label: "swa128_decode_4k",
        b: 1,
        tq: 1,
        tkv: 4096,
        h: 32,
        hkv: 8,
        d: 128,
        window: Some(1024),
        q_off: 4095,
        kv_off: 0,
    },
    Cfg {
        label: "swa128_decode_b8_4k",
        b: 8,
        tq: 1,
        tkv: 4096,
        h: 32,
        hkv: 8,
        d: 128,
        window: Some(1024),
        q_off: 4095,
        kv_off: 0,
    },
    // Two larger batches, to find where the KV-split kernel stops winning.
    // `ATTN_SPLIT_KV_BELOW_TG` was picked when the only decode configs above
    // the threshold was `b8` at B*H = 256, and picked from a measurement that
    // was ~85% command-buffer submit. Kernel-only, the split kernel is still
    // 1.8x ahead there, so the crossover -- if there is one -- is further out
    // than the config set could see.
    Cfg {
        label: "swa128_decode_b32_1k",
        b: 32,
        tq: 1,
        tkv: 1024,
        h: 32,
        hkv: 8,
        d: 128,
        window: Some(1024),
        q_off: 1023,
        kv_off: 0,
    },
    Cfg {
        label: "swa128_decode_b64_1k",
        b: 64,
        tq: 1,
        tkv: 1024,
        h: 32,
        hkv: 8,
        d: 128,
        window: Some(1024),
        q_off: 1023,
        kv_off: 0,
    },
    // ---- D=256 sliding window ----
    Cfg {
        label: "swa256_prefill_2048",
        b: 1,
        tq: 2048,
        tkv: 2048,
        h: 16,
        hkv: 4,
        d: 256,
        window: Some(1024),
        q_off: 0,
        kv_off: 0,
    },
    Cfg {
        label: "swa256_decode_4k",
        b: 1,
        tq: 1,
        tkv: 4096,
        h: 16,
        hkv: 4,
        d: 256,
        window: Some(1024),
        q_off: 4095,
        kv_off: 0,
    },
    // ---- D=512 global causal ----
    Cfg {
        label: "global512_prefill_1024",
        b: 1,
        tq: 1024,
        tkv: 1024,
        h: 8,
        hkv: 2,
        d: 512,
        window: None,
        q_off: 0,
        kv_off: 0,
    },
    Cfg {
        label: "global512_decode_4k",
        b: 1,
        tq: 1,
        tkv: 4096,
        h: 8,
        hkv: 2,
        d: 512,
        window: None,
        q_off: 4095,
        kv_off: 0,
    },
    // Same shape with no GQA, to separate *issued* K/V traffic from *unique*.
    // At 4:1 the four query heads sharing a KV head each re-read it, so the
    // kernel issues 4x the unique bytes and the cache absorbs the difference.
    // This config issues the same bytes and reads four times as many unique
    // ones, which is what says whether the ceiling is on the load path or on
    // DRAM.
    Cfg {
        label: "global512_decode_4k_mha",
        b: 1,
        tq: 1,
        tkv: 4096,
        h: 8,
        hkv: 8,
        d: 512,
        window: None,
        q_off: 4095,
        kv_off: 0,
    },
    // Hkv = 1 makes the [B, Tkv, Hkv, D] layout *already* contiguous along the
    // key axis, so this is the contiguous-stream case with no layout change:
    // whatever a head-major cache would buy, this config already has.
    Cfg {
        label: "global512_decode_4k_mqa",
        b: 1,
        tq: 1,
        tkv: 4096,
        h: 8,
        hkv: 1,
        d: 512,
        window: None,
        q_off: 4095,
        kv_off: 0,
    },
];

fn u32_buf(rt: &Arc<GpuRuntime>, v: u32) -> Result<GpuBuffer, String> {
    let b = rt.alloc_buffer(4)?;
    b.write_u32(&[v]);
    Ok(b)
}

/// The seven buffers every attention dispatch needs. Bundled so the launch and
/// timing helpers take one operand set rather than a seven-argument tail whose
/// order is easy to transpose silently.
struct Bufs<'a> {
    q: &'a GpuBuffer,
    k: &'a GpuBuffer,
    v: &'a GpuBuffer,
    o: &'a GpuBuffer,
    tkv: &'a GpuBuffer,
    qo: &'a GpuBuffer,
    ko: &'a GpuBuffer,
}

/// Which implementation a lane measures.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Impl {
    /// The original BR-tiled kernels, kept as the A/B baseline.
    Tiled,
    /// What the public entry points actually dispatch, routing by Tq.
    Routed,
    /// FlashDecoding: split over KV, `Tq == 1` only.
    Decode,
    /// Row-parallel: one simdgroup per query row.
    Rows,
}

impl Impl {
    fn tag(self) -> &'static str {
        match self {
            Impl::Tiled => "tessl-tiled",
            Impl::Routed => "tessl",
            Impl::Decode => "tessl-decode",
            Impl::Rows => "tessl-rows",
        }
    }
}

/// Benchmark controls parsed once, before Metal is initialized.
///
/// These used to be read from `std::env` inside [`launch_impl`], which put six
/// environment lookups and string parses inside every timed kernel launch.
/// Keeping the immutable result here makes the measurement path describe the
/// dispatch rather than the host configuration parser.
#[derive(Clone, Copy)]
struct Tuning {
    batch: usize,
    include_all_impls: bool,
    decode_lanes: Option<nn::RowsLanes>,
    decode_head_block: Option<nn::DecodeHeadBlock>,
    reduce_width: Option<usize>,
    rows_groups: Option<nn::RowsGroups>,
    decode_chunk: Option<nn::DecodeChunk>,
    rows_lanes: Option<nn::RowsLanes>,
}

impl Tuning {
    fn from_env() -> Result<Self, String> {
        Ok(Self {
            batch: batch_size()?,
            include_all_impls: include_all_impls()?,
            decode_lanes: optional_tuning("BENCH_ATTN_DECODE_R", nn::RowsLanes::parse)?,
            decode_head_block: optional_tuning(
                "BENCH_ATTN_DECODE_SGS",
                nn::DecodeHeadBlock::parse,
            )?,
            reduce_width: optional_tuning("BENCH_ATTN_REDUCE_W", parse_reduce_width)?,
            rows_groups: optional_tuning("BENCH_ATTN_ROWS_SGT", nn::RowsGroups::parse)?,
            decode_chunk: optional_tuning("BENCH_ATTN_DECODE_CHUNK", nn::DecodeChunk::parse)?,
            rows_lanes: optional_tuning("BENCH_ATTN_ROWS_R", nn::RowsLanes::parse)?,
        })
    }

    fn decode_lanes(self, c: &Cfg) -> nn::RowsLanes {
        self.decode_lanes
            .unwrap_or_else(|| nn::decode_lanes_for(c.d as u32))
    }

    fn decode_chunk(self, c: &Cfg) -> nn::DecodeChunk {
        self.decode_chunk
            .unwrap_or_else(|| nn::decode_chunk_for(c.d as u32))
    }

    fn rows_lanes(self, c: &Cfg) -> nn::RowsLanes {
        self.rows_lanes
            .unwrap_or_else(|| nn::rows_lanes_for(c.d as u32))
    }

    fn rows_groups(self, c: &Cfg) -> nn::RowsGroups {
        self.rows_groups
            .unwrap_or_else(|| nn::rows_groups_for(c.d as u32))
    }
}

fn optional_tuning<T>(
    name: &str,
    parse: impl FnOnce(&str) -> Result<T, String>,
) -> Result<Option<T>, String> {
    match std::env::var(name) {
        Ok(value) => parse(value.trim())
            .map(Some)
            .map_err(|e| format!("{name}: {e}")),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(format!("{name}: {e}")),
    }
}

fn parse_reduce_width(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(n) if (32..=1024).contains(&n) && n % 32 == 0 => Ok(n),
        _ => Err(format!(
            "must be a multiple of 32 in [32, 1024], got {value:?}"
        )),
    }
}

/// One dispatch of whichever kernel this configuration selects.
/// The kernel an implementation dispatches for this config.
///
/// One owner, because the timing rows and the parity manifest both name it and
/// two copies of the match would drift.
fn kernel_name(imp: Impl, c: &Cfg) -> Result<&'static str, String> {
    Ok(match (imp, c.window) {
        (Impl::Decode, _) => "flash_attn_decode",
        (Impl::Rows, _) => "flash_attn_rows",
        (Impl::Routed, _) => "routed",
        (Impl::Tiled, Some(_)) => c.head_dim()?.kernel(),
        (Impl::Tiled, None) => "flash_attn_global_h512",
    })
}

fn launch_impl(
    rt: &Arc<GpuRuntime>,
    c: &Cfg,
    b: &Bufs,
    imp: Impl,
    tuning: Tuning,
) -> Result<(), String> {
    if imp == Impl::Decode {
        return nn::flash_attn_decode_with_chunk(
            rt,
            b.q,
            b.k,
            b.v,
            b.o,
            b.tkv,
            b.qo,
            b.ko,
            c.dims(),
            c.d as u32,
            c.tkv,
            tuning.decode_chunk(c),
            tuning.decode_lanes(c),
            tuning.reduce_width,
            tuning.decode_head_block,
            false,
        );
    }
    if imp == Impl::Routed {
        // The public entry points, exactly as a caller reaches them.
        return match c.window {
            Some(_) => nn::flash_attn_swa(
                rt,
                c.head_dim()?,
                b.q,
                b.k,
                b.v,
                b.o,
                b.tkv,
                b.qo,
                b.ko,
                c.dims(),
            ),
            None => nn::flash_attn_global_h512(
                rt,
                b.q,
                b.k,
                b.v,
                b.o,
                b.tkv,
                b.qo,
                b.ko,
                c.dims(),
                false,
            ),
        };
    }
    if imp == Impl::Rows {
        return nn::flash_attn_rows_with_lanes(
            rt,
            b.q,
            b.k,
            b.v,
            b.o,
            b.tkv,
            b.qo,
            b.ko,
            c.dims(),
            c.d as u32,
            tuning.rows_lanes(c),
            tuning.rows_groups(c),
            false,
        );
    }
    launch(rt, c, b)
}

fn launch(rt: &Arc<GpuRuntime>, c: &Cfg, b: &Bufs) -> Result<(), String> {
    match c.window {
        Some(_) => nn::flash_attn_swa_tiled(
            rt,
            c.head_dim()?,
            b.q,
            b.k,
            b.v,
            b.o,
            b.tkv,
            b.qo,
            b.ko,
            c.dims(),
        ),
        None => nn::flash_attn_global_h512_tiled(
            rt,
            b.q,
            b.k,
            b.v,
            b.o,
            b.tkv,
            b.qo,
            b.ko,
            c.dims(),
            false,
        ),
    }
}

/// Launches per synchronize.
///
/// `1` is submit-and-wait per call, which is what the cross-runtime comparison
/// uses because it is what `mx.eval` and `torch.mps.synchronize` do. Anything
/// larger amortises the host submit across N launches and isolates the kernel
/// from it: on this machine a *trivial* elementwise kernel measures 4 us
/// batched and 178 us solo, so at decode sizes the solo number is almost
/// entirely round-trip. A real decode loop pays that round trip once for a
/// whole model step, not once per attention call.
fn batch_size() -> Result<usize, String> {
    env_usize("BENCH_ATTN_BATCHED", 1, 1)
}

fn include_all_impls() -> Result<bool, String> {
    match std::env::var("BENCH_ATTN_IMPLS") {
        Ok(v) => parse_impl_selection(Some(&v)),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(e) => Err(format!("BENCH_ATTN_IMPLS: {e}")),
    }
}

fn parse_impl_selection(raw: Option<&str>) -> Result<bool, String> {
    match raw {
        Some(v) if v.trim() == "all" => Ok(true),
        Some(v) => Err(format!(
            "BENCH_ATTN_IMPLS={v:?} is not supported; expected \"all\" or unset"
        )),
        None => Ok(false),
    }
}

fn parse_requested_configs(raw: Option<&str>) -> Result<Option<Vec<String>>, String> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let requested = raw
        .split(',')
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect::<Vec<_>>();
    if requested.is_empty() {
        return Err("BENCH_ATTN_CFGS is set but names no configurations".to_string());
    }
    Ok(Some(requested))
}

fn time_cfg(
    rt: &Arc<GpuRuntime>,
    c: &Cfg,
    b: &Bufs,
    imp: Impl,
    tuning: Tuning,
    warmup: usize,
    iters: usize,
) -> Result<Vec<f64>, String> {
    let batch = tuning.batch;
    // Without this every dispatch gets its own command buffer and commits, so a
    // loop of `batch` launches costs `batch` submits and the batched arm
    // silently measures the same thing as solo. `bench_nn_kernels` carries the
    // same warning; this benchmark reproduced the bug it describes before the
    // flag was set here.
    if batch > 1 {
        rt.set_async_encode(true)?;
    }
    for _ in 0..warmup {
        for _ in 0..batch {
            launch_impl(rt, c, b, imp, tuning)?;
        }
        rt.synchronize()?;
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        for _ in 0..batch {
            launch_impl(rt, c, b, imp, tuning)?;
        }
        rt.synchronize()?;
        samples.push(t0.elapsed().as_secs_f64() * 1000.0 / batch as f64);
    }
    if batch > 1 {
        rt.set_async_encode(false)?;
    }
    Ok(samples)
}

/// Sentinel the output is seeded with, so a kernel that leaves rows untouched
/// is caught rather than inheriting whatever the allocator handed back.
const UNWRITTEN: f32 = -6.5e28;

/// Emit the kernel trace for `bench/kernel_coverage.py`. Prints nothing when
/// tracing is off, so normal runs are unchanged.
fn emit_kernel_trace() {
    if tessl::runtime::kernel_trace_enabled() {
        eprintln!(
            "KERNEL_TRACE {}",
            tessl::runtime::traced_kernels().join(",")
        );
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dump_dir = match args.iter().position(|a| a == "--dump-parity") {
        Some(i) => Some(
            args.get(i + 1)
                .filter(|v| !v.starts_with("--"))
                .ok_or("--dump-parity requires a directory argument")?
                .clone(),
        ),
        None => None,
    };
    // A parity artifact describes the configuration the library ships. Under a
    // tuning override it would describe a kernel no caller reaches, while the
    // manifest named the shipping one -- so the override is refused rather than
    // recorded. `bench_gemm_sweep` rejects `BENCH_SHAPES` beside `--dump-parity`
    // for the same reason.
    if dump_dir.is_some() {
        for var in [
            "BENCH_ATTN_DECODE_CHUNK",
            "BENCH_ATTN_DECODE_R",
            "BENCH_ATTN_ROWS_R",
            "BENCH_ATTN_ROWS_SGT",
            "BENCH_ATTN_REDUCE_W",
            "BENCH_ATTN_DECODE_SGS",
            "TESSL_ATTN_TILED",
        ] {
            if std::env::var_os(var).is_some() {
                return Err(format!(
                    "{var} is set alongside --dump-parity. A parity dump must \
                     describe the shipping configuration; sweep tuning values \
                     with bench/attn_tune.py instead."
                ));
            }
        }
    }

    let warmup = env_usize("BENCH_WARMUP", 10, 0)?;
    let iters = env_usize("BENCH_ITERS", 50, 1)?;
    let tuning = Tuning::from_env()?;
    let dist = match std::env::var("BENCH_ATTN_DIST") {
        Ok(v) => Dist::parse(v.trim())?,
        Err(std::env::VarError::NotPresent) => Dist::Uniform,
        Err(e) => return Err(format!("BENCH_ATTN_DIST: {e}")),
    };
    let only = match std::env::var("BENCH_ATTN_CFGS") {
        Ok(v) => parse_requested_configs(Some(&v))?,
        Err(std::env::VarError::NotPresent) => None,
        Err(e) => return Err(format!("BENCH_ATTN_CFGS: {e}")),
    };

    let rt = GpuRuntime::new()?;
    let cfgs: Vec<&Cfg> = match &only {
        None => CFGS.iter().collect(),
        Some(want) => {
            let known: Vec<&str> = CFGS.iter().map(|c| c.label).collect();
            for w in want {
                if !known.contains(&w.as_str()) {
                    return Err(format!(
                        "BENCH_ATTN_CFGS: {w:?} is not a configuration; expected from {known:?}"
                    ));
                }
            }
            CFGS.iter()
                .filter(|c| want.contains(&c.label.to_string()))
                .collect()
        }
    };

    if let Some(dir) = &dump_dir {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {dir}: {e}"))?;
    }

    let mut rows: Vec<String> = Vec::new();
    let mut dumped: Vec<String> = Vec::new();
    for c in cfgs {
        let qh = fill_dist(c.q_elems(), 1, dist);
        let kh = fill_dist(c.kv_elems(), 2, dist);
        let vh = fill_dist(c.kv_elems(), 3, dist);
        let q = rt.alloc_buffer(qh.len() * 4)?;
        let k = rt.alloc_buffer(kh.len() * 4)?;
        let v = rt.alloc_buffer(vh.len() * 4)?;
        let o = rt.alloc_buffer(c.q_elems() * 4)?;
        q.write_f32(&qh);
        k.write_f32(&kh);
        v.write_f32(&vh);
        o.write_f32(&vec![UNWRITTEN; c.q_elems()]);

        let tkv = u32_buf(&rt, c.tkv as u32)?;
        let qo = u32_buf(&rt, c.q_off as u32)?;
        let ko = u32_buf(&rt, c.kv_off as u32)?;

        let bufs = Bufs {
            q: &q,
            k: &k,
            v: &v,
            o: &o,
            tkv: &tkv,
            qo: &qo,
            ko: &ko,
        };
        // Every Tq == 1 config is timed on both implementations in the same
        // run, so the comparison cannot pick up drift between two invocations.
        // The tiled baseline is 7-25x slower, so at batch>1 it dominates the
        // run time while contributing nothing: the batched arm exists to
        // isolate the *fast* kernels from submit cost. `BENCH_ATTN_IMPLS=all`
        // forces it back in.
        let batched = tuning.batch > 1;
        let impls: &[Impl] = match (c.tq == 1, batched && !tuning.include_all_impls) {
            (true, false) => &[Impl::Tiled, Impl::Routed, Impl::Decode, Impl::Rows],
            (true, true) => &[Impl::Routed, Impl::Decode, Impl::Rows],
            (false, false) => &[Impl::Tiled, Impl::Routed, Impl::Rows],
            (false, true) => &[Impl::Routed, Impl::Rows],
        };
        for &imp in impls {
            let samples = time_cfg(&rt, c, &bufs, imp, tuning, warmup, iters)?;
            let med = median(samples.clone())?;
            if med <= 0.0 {
                return Err(format!("{}: median {med} ms is not positive", c.label));
            }
            let best = samples.iter().cloned().fold(f64::INFINITY, f64::min);
            let live = c.live_flop();
            let gflops = live / (med * 1e6);
            let kernel = kernel_name(imp, c)?;
            eprintln!(
                "{:<22} {:<13} {kernel:<28} Tkv={} H={} D={}  {med:8.3} ms  {gflops:9.1} GFLOP/s",
                c.label,
                imp.tag(),
                c.tkv,
                c.h,
                c.d
            );
            rows.push(format!(
                r#"{{"cfg":"{}","kernel":"{kernel}","runtime":"{}","batched":{},"b":{},"tq":{},"tkv":{},"h":{},"hkv":{},"d":{},"window":{},"q_off":{},"kv_off":{},"median_ms":{med:.6},"best_ms":{best:.6},"live_pairs":{},"gflops":{gflops:.3},"dense_gflops":{:.3}}}"#,
                c.label, imp.tag(), tuning.batch, c.b, c.tq, c.tkv, c.h, c.hkv, c.d,
                match c.window { Some(w) => w as i64, None => -1 },
                c.q_off, c.kv_off, c.live_pairs(),
                c.dense_flop() / (med * 1e6)
            ));
        }

        if let Some(dir) = &dump_dir {
            let d = Path::new(dir).join(c.label);
            std::fs::create_dir_all(&d).map_err(|e| format!("mkdir {}: {e}", d.display()))?;
            write_npy_f32(&d.join("q.npy"), &[c.b, c.tq, c.h, c.d], &qh)?;
            write_npy_f32(&d.join("k.npy"), &[c.b, c.tkv, c.hkv, c.d], &kh)?;
            write_npy_f32(&d.join("v.npy"), &[c.b, c.tkv, c.hkv, c.d], &vh)?;
            // Every implementation is dumped under its own name, from its own
            // dispatch. `o_tessl.npy` used to be whatever `o` happened to hold
            // when the timing loop ended -- the *last* implementation in the
            // list, which at Tq == 1 is the row kernel, while the manifest
            // named the tiled kernel beside it. The scorer then reported the
            // row kernel's error under the routed path's name. A dumped result
            // must come from the dispatch it is labelled with, and reordering
            // the timing list must not silently change what is scored.
            let mut lanes = Vec::new();
            for &imp in impls {
                o.write_f32(&vec![UNWRITTEN; c.q_elems()]);
                launch_impl(&rt, c, &bufs, imp, tuning)?;
                rt.synchronize()?;
                let out = o.read_f32()[..c.q_elems()].to_vec();
                if let Some(i) = out.iter().position(|x| *x == UNWRITTEN) {
                    return Err(format!(
                        "{}/{}: output element {i} was never written by the kernel",
                        c.label,
                        imp.tag()
                    ));
                }
                if let Some(i) = out.iter().position(|x| !x.is_finite()) {
                    return Err(format!(
                        "{}/{}: non-finite output {} at element {i}",
                        c.label,
                        imp.tag(),
                        out[i]
                    ));
                }
                write_npy_f32(
                    &d.join(format!("o_{}.npy", imp.tag())),
                    &[c.b, c.tq, c.h, c.d],
                    &out,
                )?;
                lanes.push(format!(
                    r#"{{"lane":"{}","kernel":"{}"}}"#,
                    imp.tag(),
                    kernel_name(imp, c)?
                ));
            }
            dumped.push(format!(
                r#"{{"cfg":"{}","lanes":[{}],"b":{},"tq":{},"tkv":{},"h":{},"hkv":{},"d":{},"window":{},"q_off":{},"kv_off":{},"scale":{}}}"#,
                c.label, lanes.join(","), c.b, c.tq, c.tkv, c.h, c.hkv, c.d,
                match c.window { Some(w) => w as i64, None => -1 },
                c.q_off, c.kv_off, c.scale()
            ));
        }
    }

    // Manifest last and only on success, so an interrupted dump cannot be
    // scored as a complete one.
    if let Some(dir) = &dump_dir {
        std::fs::write(
            Path::new(dir).join("attn_manifest.json"),
            format!(
                r#"{{"dist":"{}","configs":[{}]}}"#,
                dist.name(),
                dumped.join(",")
            ),
        )
        .map_err(|e| format!("write manifest: {e}"))?;
        eprintln!("attention parity dump complete: {} configs", dumped.len());
    }
    println!("[{}]", rows.join(","));
    Ok(())
}

fn main() -> Result<(), String> {
    // Wrapped rather than called before each `return`: --dump-parity exits
    // early and an error exits earlier still, and a trace that silently skips
    // those paths would under-report coverage exactly where it matters.
    let outcome = run();
    emit_kernel_trace();
    outcome
}

#[cfg(test)]
mod tests {
    use super::{parse_impl_selection, parse_reduce_width, parse_requested_configs};

    #[test]
    fn implementation_selection_rejects_silent_fallbacks() {
        assert!(!parse_impl_selection(None).unwrap());
        assert!(parse_impl_selection(Some(" all ")).unwrap());
        assert!(parse_impl_selection(Some("default")).is_err());
        assert!(parse_impl_selection(Some("")).is_err());
    }

    #[test]
    fn configuration_selection_rejects_an_empty_override() {
        assert!(parse_requested_configs(None).unwrap().is_none());
        assert_eq!(
            parse_requested_configs(Some("a, b")).unwrap().unwrap(),
            ["a", "b"]
        );
        assert!(parse_requested_configs(Some(" , ")).is_err());
    }

    #[test]
    fn reduce_width_is_warp_aligned_and_bounded() {
        assert_eq!(parse_reduce_width("32").unwrap(), 32);
        assert_eq!(parse_reduce_width("1024").unwrap(), 1024);
        for invalid in ["", "0", "31", "33", "1056", "not-a-width"] {
            assert!(parse_reduce_width(invalid).is_err(), "accepted {invalid:?}");
        }
    }

    #[test]
    fn timed_launch_path_does_not_read_the_environment() {
        let source = include_str!("bench_flash_attn.rs");
        let launch = source
            .split_once("fn launch_impl(")
            .expect("launch_impl exists")
            .1
            .split_once("\nfn launch(")
            .expect("launch_impl has a bounded source region")
            .0;
        assert!(
            !launch.contains("std::env"),
            "environment parsing inside launch_impl contaminates every timing sample"
        );
    }
}
