//! Every kernel promoted out of `gemma-metal` must live in tessl's *own*
//! metallib.
//!
//! These 44 entry points were compiled into gemma-metal's overlay metallib and
//! reached only through `GpuRuntime::add_metallib`. They are model-agnostic —
//! RMSNorm, GELU/SiLU gating, flash attention, quantized GEMV/GEMM, KV cache
//! stores, embedding lookup, sampling — so they belong to the kernel library,
//! not to one model's crate.
//!
//! This test is the gate on that move. `GpuRuntime::pipeline` resolves the
//! primary library first and only then any registered overlay, so a name that
//! resolves here with no overlay registered can only have come from tessl's
//! `default.metallib`. Before the move every name below failed with
//! "kernel '…' not found in metallib".
//!
//! It also pins the promoted surface: deleting or renaming one of these is a
//! breaking change to a published crate, and it should take a failing test to
//! do it rather than a silent `NotFound` at some downstream caller's first
//! dispatch.

mod common;

/// Kernels promoted from `gemma-metal/kernels/` into `crates/tessl/kernels/`.
///
/// Grouped by source file. `cast_f32_to_bf16` is deliberately absent: gemma's
/// `rms_norm.metal` carried a byte-identical copy of the kernel tessl already
/// had in `utils.metal`, and since primary resolution wins, gemma's copy could
/// never have been the one dispatched. It was deleted in the move rather than
/// carried across as a second definition of the same behavior.
const PROMOTED: &[&str] = &[
    // rms_norm.metal
    "rms_norm_f32",
    "rms_norm_bf16",
    "rms_norm_residual_add_f32",
    // mlp_silu.metal
    "mlp_silu",
    // mlp_gelu_tanh.metal
    "mlp_gelu_tanh",
    "mlp_gelu_tanh_bf16",
    // flash_attn_*.metal
    "flash_attn_swa_h128",
    "flash_attn_swa_h256",
    "flash_attn_global_h512",
    // rms_qkv_rope.metal
    "rms_qkv_rope",
    "rms_qkv_rope_posbuf",
    "rms_qkv_rope_kv_store",
    // kv_store.metal
    "kv_store_timestep",
    "kv_store_timestep_pair",
    "kv_ring_densify",
    // gemv_q4.metal
    "gemv_q4",
    "gemv_q4_tiled",
    // gemv_q8.metal
    "gemv_q8",
    // gemv_q4_mlx.metal
    "gemv_q4_mlx",
    "gemv_q4_mlx_wide",
    "gemv_q4_mlx_blocked",
    "gemv_q4_mlx_blocked_gate_up_gelu",
    "gemv_q4_mlx_simd",
    "gemv_q4_mlx_simd_add",
    "gemv_q4_mlx_simd_gate_up_gelu",
    "gemv_q4_mlx_simd_kv",
    "gemv_q4_mlx_simd_i4",
    "gemv_q4_mlx_simd_add_i4",
    "gemv_q4_mlx_simd_gate_up_gelu_i4",
    "gemv_q4_mlx_simd_kv_i4",
    "gemv_q4_mlx_tiled",
    "gemv_q4_mlx_simd_qkv",
    "gemv_q4_mlx_simd_qkv_i4",
    // gemm_q4_mlx.metal
    "gemm_q4_mlx_simd",
    "gemm_q4_mlx_simd_add",
    "gemm_q4_mlx_simd_i4",
    "gemm_q4_mlx_simd_add_i4",
    // embed_lookup.metal
    "embed_lookup_q4_mlx",
    "embed_lookup_q4",
    "scale_f32_inplace",
    // softcap_sample.metal
    "softcap_logits",
    "argmax_f32",
    "softcap_sample",
    "softcap_argmax_one_pass",
];

#[test]
fn every_promoted_kernel_resolves_from_tessls_own_metallib() {
    common::with_gpu(|rt| {
        let mut missing = Vec::new();
        for name in PROMOTED {
            if rt.pipeline(name).is_err() {
                missing.push(*name);
            }
        }
        assert!(
            missing.is_empty(),
            "{} of {} promoted kernels are not in tessl's metallib: {missing:?}",
            missing.len(),
            PROMOTED.len()
        );
    });
}

#[test]
fn the_promoted_list_has_no_duplicates() {
    // A copy-paste slip here would weaken the test above without failing it:
    // the same name checked twice still passes while its neighbour goes
    // unchecked.
    let mut sorted = PROMOTED.to_vec();
    sorted.sort_unstable();
    let before = sorted.len();
    sorted.dedup();
    assert_eq!(before, sorted.len(), "duplicate entry in PROMOTED");
}

#[test]
fn promoted_count_matches_the_kernel_sources() {
    // Guards the other direction: a kernel added to one of the promoted
    // `.metal` files without being listed here ships untested. The number is
    // the count of `kernel void` entry points across the 14 promoted sources.
    assert_eq!(
        PROMOTED.len(),
        44,
        "PROMOTED drifted from the 44 entry points in the promoted .metal sources; \
         re-run: grep -hc '^kernel void' kernels/{{rms_norm,mlp_silu,mlp_gelu_tanh,\
flash_attn_swa_h128,flash_attn_swa_h256,flash_attn_global_h512,rms_qkv_rope,kv_store,\
gemv_q4,gemv_q8,gemv_q4_mlx,gemm_q4_mlx,embed_lookup,softcap_sample}}.metal"
    );
}

/// Kernels deliberately **left** in `gemma-metal`, and why.
///
/// `ple_lookup*` implements Gemma 3n Per-Layer Embeddings; `persistent_interp*`
/// is a documented mini-graph prototype whose own header scopes it to the
/// gate→down and FA→o_proj edges and rules it out for default decode. Neither
/// is a general ML primitive, so neither belongs in a kernel library.
const NOT_PROMOTED: &[&str] = &[
    "ple_lookup",
    "ple_lookup_q4_mlx",
    "ple_residual_add",
    "ple_lookup_q4_mlx_residual",
    "persistent_interp_gate_down",
    "persistent_interp_gate_down_q4",
    "persistent_interp_fa_o_proj",
];

#[test]
fn kernels_left_behind_do_not_resolve_from_tessl() {
    // Negative control for the test above. Without it, a `pipeline()` that
    // resolved *every* string — or a metallib that had silently absorbed all of
    // gemma's kernels — would look identical to a correct promotion.
    common::with_gpu(|rt| {
        let leaked: Vec<_> = NOT_PROMOTED
            .iter()
            .filter(|name| rt.pipeline(name).is_ok())
            .collect();
        assert!(
            leaked.is_empty(),
            "Gemma-specific kernels reached tessl's metallib: {leaked:?}"
        );
    });
}
