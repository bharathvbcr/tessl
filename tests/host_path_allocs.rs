//! What the per-dispatch host path allocates on success: nothing, where a
//! decode loop calls it every token.
//!
//! A counting global allocator, armed per thread, counts every Rust heap
//! allocation between `arm` and `disarm`. The pipeline-mode flag is
//! process-wide, which is why this is its own test binary and every check
//! runs in one test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
}

fn note() {
    if ARMED.with(Cell::get) {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: every method forwards to `System` unchanged; the counter is a side
// effect that neither allocates nor touches the returned memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Heap allocations `f` makes on this thread.
fn allocations<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCS.load(Ordering::Relaxed);
    ARMED.with(|a| a.set(true));
    let out = f();
    ARMED.with(|a| a.set(false));
    (out, ALLOCS.load(Ordering::Relaxed) - before)
}

struct IcbModeRestore;

impl Drop for IcbModeRestore {
    fn drop(&mut self) {
        tessl::decode_icb::set_icb_pipelines(false);
    }
}

#[test]
fn a_pipeline_cache_hit_allocates_nothing_in_either_mode() {
    let rt = tessl::GpuRuntime::new().expect("GpuRuntime::new");
    let _restore = IcbModeRestore;
    for icb in [false, true] {
        tessl::decode_icb::set_icb_pipelines(icb);
        rt.pipeline("rms_norm_f32").expect("warm the cache");
        let (hits, n) = allocations(|| (0..64).map(|_| rt.pipeline("rms_norm_f32")).collect::<Vec<_>>());
        // The Vec holding the 64 results is the one allocation allowed.
        assert_eq!(n, 1, "icb = {icb}: {} allocations over 64 cache hits", n - 1);
        assert!(hits.iter().all(Result::is_ok));
    }
}

/// Every entry point a batch-1 Qwen3.5 decode token encodes, at small shapes,
/// with async encode on: what each call allocates on the host on success.
/// Validation, pipeline lookup and scratch are all on this path, every layer
/// of every token.
#[test]
fn decode_path_entry_points_allocate_nothing_on_success() {
    use tessl::qwen35::{
        self, AttnProjLayout, AttnShape, AttnTargets, Cols, GdnParams, GdnProjLayout, OutCols, SharedPrefix, StateIn,
    };
    use tessl::{gemm, gemm_epilogue, nn, DType, Epilogue, GemmBackend};

    let rt = &tessl::GpuRuntime::new().expect("GpuRuntime::new");
    if !rt.has_tensorops() {
        eprintln!("skipped: the bf16 GEMMs need TensorOps");
        return;
    }
    const H: usize = 128;
    const INTER: usize = 256;
    const KW: u32 = 4;
    const P: usize = 64;
    const S_CAP: usize = 8;
    let gl = GdnProjLayout::new(1, 2, 64).unwrap();
    let al = AttnProjLayout::new(2, 1, 256).unwrap();
    let f32b = |n: usize, v: f32| {
        let b = rt.alloc_buffer(n.max(1) * 4).unwrap();
        b.write_f32(&vec![v; n.max(1)]);
        b
    };
    let u32b = |v: u32| {
        let b = rt.alloc_buffer(4).unwrap();
        b.write_u32(&[v]);
        b
    };
    let bf16 = |r: usize, c: usize| rt.alloc_tensor_bf16(&[r, c]).unwrap();
    let f32t = |r: usize, c: usize| rt.alloc_tensor_f32(&[r, c]).unwrap();

    let resid = f32t(1, H);
    let norm_w = f32b(H, 0.0);
    let xb = bf16(1, H);
    let g_w = bf16(H, gl.width() as usize);
    let g_proj = f32t(1, gl.width() as usize);
    let conv_w = f32b(gl.conv_dim() as usize * KW as usize, 0.1);
    let conv_state = [
        f32b(gl.conv_dim() as usize * 3, 0.0),
        f32b(gl.conv_dim() as usize * 3, 0.0),
    ];
    let qkv = f32b(gl.conv_dim() as usize, 0.0);
    let (a_log, dt_bias) = (f32b(2, 0.5), f32b(2, -3.0));
    let gdn_state = f32b(gl.dims(1, 1).state_elems_per_row(), 0.0);
    let g_o = f32b(gl.value_dim() as usize, 0.0);
    let g_norm = f32b(64, 1.0);
    let g_y = bf16(1, gl.value_dim() as usize);
    let g_out_w = bf16(gl.value_dim() as usize, H);
    let a_proj = f32t(1, al.width() as usize);
    let (q_norm, k_norm) = (f32b(256, 0.0), f32b(256, 0.0));
    let a_q = f32b(2 * 256, 0.0);
    let (pk, pv) = (f32b(P * 256, 0.1), f32b(P * 256, 0.1));
    let (sk, sv) = (f32b(S_CAP * 256, 0.0), f32b(S_CAP * 256, 0.0));
    let (q_pos, s_len) = (u32b(P as u32), u32b(1));
    let a_o = f32b(2 * 256, 0.0);
    let a_y = bf16(1, 2 * 256);
    let scratch = nn::DecodeScratch::new(rt, 1, 2, P + S_CAP, 256).unwrap();
    let (m_gate, m_up) = (f32t(1, INTER), f32t(1, INTER));
    let m_mid = bf16(1, INTER);
    let logits = f32b(1000, 0.5);
    let (cap, tok) = (f32b(1, 30.0), u32b(0));
    let (idx, val) = (rt.alloc_buffer(16).unwrap(), f32b(4, 0.0));
    let tkv = u32b(P as u32);
    let zero = u32b(0);
    let attn_dims = nn::AttnDims {
        batch: 1,
        tq: 1,
        heads: 2,
        heads_kv: 1,
        window: 0,
        scale: 0.0625,
    };
    let backend = GemmBackend::TensorOps;
    let resid_add = Epilogue {
        beta: 1.0,
        ..Epilogue::default()
    };
    rt.synchronize().unwrap();

    type Step<'a> = (&'static str, Box<dyn Fn() -> Result<(), String> + 'a>);
    let steps: Vec<Step<'_>> = vec![
        (
            "nn::rms_norm_bf16",
            Box::new(|| nn::rms_norm_bf16(rt, &resid.buffer, &norm_w, &xb.buffer, 1, H as u32, 1e-6)),
        ),
        ("gemm (in-proj)", Box::new(|| gemm(&xb, &g_w, &g_proj, backend))),
        (
            "qwen35::conv1d_silu",
            Box::new(|| {
                qwen35::conv1d_silu(
                    rt,
                    Cols::dense(&g_proj.buffer, gl.width()),
                    &conv_w,
                    KW,
                    StateIn::PerBatch(&conv_state[0]),
                    &qkv,
                    Some(&conv_state[1]),
                    1,
                    1,
                    gl.conv_dim(),
                )
            }),
        ),
        (
            "qwen35::gdn_recurrent",
            Box::new(|| {
                qwen35::gdn_recurrent(
                    rt,
                    &gl.dims(1, 1),
                    &gl.conv_qkv(&qkv),
                    &gl.gates(&g_proj.buffer),
                    &GdnParams {
                        a_log: &a_log,
                        dt_bias: &dt_bias,
                    },
                    StateIn::PerBatch(&gdn_state),
                    Cols::dense(&g_o, gl.value_dim()),
                    Some(&gdn_state),
                )
            }),
        ),
        (
            "qwen35::gated_rms_norm",
            Box::new(|| {
                qwen35::gated_rms_norm(
                    rt,
                    Cols::dense(&g_o, gl.value_dim()),
                    gl.z(&g_proj.buffer),
                    &g_norm,
                    OutCols {
                        cols: Cols::dense(&g_y.buffer, gl.value_dim()),
                        dtype: DType::BF16,
                    },
                    1,
                    2,
                    64,
                    1e-6,
                )
            }),
        ),
        (
            "gemm_epilogue (out-proj)",
            Box::new(|| gemm_epilogue(&g_y, &g_out_w, &resid, backend, resid_add)),
        ),
        (
            "qwen35::attn_qk_norm_rope_suffix_posbuf",
            Box::new(|| {
                qwen35::attn_qk_norm_rope_suffix_posbuf(
                    rt,
                    &AttnShape {
                        batch: 1,
                        seq: 1,
                        q_heads: 2,
                        kv_heads: 1,
                        head_dim: 256,
                        rotary_dim: 64,
                    },
                    Cols::dense(&a_proj.buffer, al.width()),
                    &q_norm,
                    &k_norm,
                    &AttnTargets {
                        q_out: &a_q,
                        k_cache: &sk,
                        v_cache: &sv,
                    },
                    P as u32,
                    &q_pos,
                    1e7,
                    1e-6,
                )
            }),
        ),
        (
            "qwen35::attn_prefix_decode",
            Box::new(|| {
                qwen35::attn_prefix_decode(
                    rt,
                    &a_q,
                    SharedPrefix {
                        k: &pk,
                        v: &pv,
                        len: P as u32,
                    },
                    &sk,
                    &sv,
                    &s_len,
                    &q_pos,
                    &a_o,
                    &scratch,
                    attn_dims,
                    false,
                )
            }),
        ),
        (
            "qwen35::attn_output_gate",
            Box::new(|| {
                qwen35::attn_output_gate(
                    rt,
                    &a_o,
                    Cols::dense(&a_proj.buffer, al.width()),
                    OutCols {
                        cols: Cols::dense(&a_y.buffer, 512),
                        dtype: DType::BF16,
                    },
                    1,
                    2,
                    256,
                )
            }),
        ),
        (
            "qwen35::swiglu",
            Box::new(|| {
                qwen35::swiglu(
                    rt,
                    Cols::dense(&m_gate.buffer, INTER as u32),
                    Cols::dense(&m_up.buffer, INTER as u32),
                    OutCols {
                        cols: Cols::dense(&m_mid.buffer, INTER as u32),
                        dtype: DType::BF16,
                    },
                    1,
                    INTER as u32,
                )
            }),
        ),
        (
            "nn::flash_attn_decode",
            Box::new(|| {
                nn::flash_attn_decode(
                    rt, &a_q, &pk, &pv, &a_o, &scratch, &tkv, &zero, &zero, attn_dims, 256, P, false,
                )
            }),
        ),
        (
            "nn::softcap_argmax_one_pass",
            Box::new(|| nn::softcap_argmax_one_pass(rt, &logits, &tok, &cap, 1000)),
        ),
        (
            "nn::argmax_f32_pass",
            Box::new(|| nn::argmax_f32_pass(rt, &logits, &idx, &val, None, &cap, 1000)),
        ),
    ];

    rt.set_async_encode(true).unwrap();
    // Warm every pipeline and the encoder first: a miss compiles, which
    // allocates, and is paid once per process, not per token.
    for (name, step) in &steps {
        step().unwrap_or_else(|e| panic!("{name}: {e}"));
    }
    let mut report = Vec::new();
    for (name, step) in &steps {
        let (result, n) = allocations(step);
        result.unwrap_or_else(|e| panic!("{name}: {e}"));
        report.push((*name, n));
    }
    rt.synchronize().unwrap();
    rt.set_async_encode(false).unwrap();
    let allocating: Vec<String> = report
        .iter()
        .filter(|(_, n)| *n > 0)
        .map(|(name, n)| format!("{name}: {n}"))
        .collect();
    assert!(
        allocating.is_empty(),
        "host allocations per decode call: {allocating:?}"
    );
}
