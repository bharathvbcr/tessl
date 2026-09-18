//! Host-only contracts for shader index arithmetic at the `u32` boundary.
//!
//! These tests deliberately inspect source rather than launching Metal work:
//! exercising the failing shapes would require multi-gigabyte allocations, and
//! a wrapped strided loop can hang a GPU instead of returning a test failure.

const DISPATCH_RS: &str = include_str!("../src/dispatch.rs");
const GEMM_RS: &str = include_str!("../src/gemm.rs");
const EMBED: &str = include_str!("../kernels/embed_lookup.metal");
const GEMV_Q4: &str = include_str!("../kernels/gemv_q4.metal");
const GEMV_Q4_MLX: &str = include_str!("../kernels/gemv_q4_mlx.metal");
const GEMM_Q4_MLX: &str = include_str!("../kernels/gemm_q4_mlx.metal");
const GEMV_Q8: &str = include_str!("../kernels/gemv_q8.metal");
const REDUCE: &str = include_str!("../kernels/reduce.metal");
const RMS_NORM: &str = include_str!("../kernels/rms_norm.metal");
const SOFTCAP_SAMPLE: &str = include_str!("../kernels/softcap_sample.metal");
const SIMD_GEMM: &str = include_str!("../kernels/matmul_simdgroup.metal");
const UTILS: &str = include_str!("../kernels/utils.metal");
const KV_STORE: &str = include_str!("../kernels/kv_store.metal");
const QKV_ROPE: &str = include_str!("../kernels/rms_qkv_rope.metal");
const ATTN_DECODE: &str = include_str!("../kernels/flash_attn_decode.metal");
const ATTN_ROWS: &str = include_str!("../kernels/flash_attn_rows.metal");
const ATTN_SWA_128: &str = include_str!("../kernels/flash_attn_swa_h128.metal");
const ATTN_SWA_256: &str = include_str!("../kernels/flash_attn_swa_h256.metal");
const ATTN_GLOBAL: &str = include_str!("../kernels/flash_attn_global_h512.metal");

/// Every `.metal` file this suite inspects.
///
/// Paired with `EXEMPT_KERNELS` and checked against the directory by
/// `every_kernel_source_is_inspected_or_explicitly_exempt`, so a new kernel
/// cannot join the build without someone deciding which list it belongs on.
const INSPECTED_KERNELS: &[&str] = &[
    "embed_lookup.metal",
    "flash_attn_decode.metal",
    "flash_attn_global_h512.metal",
    "flash_attn_rows.metal",
    "flash_attn_swa_h128.metal",
    "flash_attn_swa_h256.metal",
    "gemm_q4_mlx.metal",
    "gemv_q4.metal",
    "gemv_q4_mlx.metal",
    "gemv_q8.metal",
    "kv_store.metal",
    "matmul_simdgroup.metal",
    "reduce.metal",
    "rms_norm.metal",
    "rms_qkv_rope.metal",
    "softcap_sample.metal",
    "utils.metal",
];

/// Kernels deliberately not inspected here, each with the reason it is safe.
const EXEMPT_KERNELS: &[(&str, &str)] = &[
    (
        "matmul_tensorops.metal",
        "TensorOps addresses through MTLTensor descriptors and `uint` tile \
         arithmetic. That is bounded by the host contract asserted in \
         `utility_and_simdgroup_pointer_math_is_explicitly_wide`: every public \
         GEMM operand stays under `i32::MAX` elements, which is below \
         `u32::MAX`. The only raw pointer maths is the batch stride, which is \
         widened.",
    ),
    (
        "tune/matmul_tensorops_tune.metal",
        "Tile-tuning variants of matmul_tensorops.metal, built only by the \
         tuning binaries and never linked into the shipped metallib; same \
         addressing and same host bound.",
    ),
    (
        "mlp_gelu_tanh.metal",
        "Elementwise over a 1D grid: the only index is the thread id, whose \
         representability is the `dispatch.rs` guard already required below.",
    ),
    (
        "mlp_silu.metal",
        "Elementwise over a 1D grid, as mlp_gelu_tanh.metal.",
    ),
];

fn require(source: &str, fragment: &str, label: &str) {
    assert!(
        source.contains(fragment),
        "{label} lost widened shader arithmetic: missing {fragment:?}"
    );
}

fn forbid(source: &str, fragment: &str, label: &str) {
    assert!(
        !source.contains(fragment),
        "{label} reintroduced narrow shader arithmetic: found {fragment:?}"
    );
}

#[test]
fn row_major_quantized_offsets_cross_u32_without_wrapping() {
    let row = 1u32 << 16;
    let stride = 1u32 << 16;
    assert_eq!(row.wrapping_mul(stride), 0);
    assert_eq!(u64::from(row) * u64::from(stride), 1u64 << 32);

    require(
        GEMV_Q4,
        "const ulong row_base = (ulong)row * cols;",
        "Q4 GEMV",
    );
    require(
        GEMV_Q4_MLX,
        "const ulong scale_base = (ulong)row * groups_per_row;",
        "MLX Q4 GEMV",
    );
    require(
        GEMV_Q8,
        "const ulong gi = (ulong)row * groups_per_row + g;",
        "Q8 GEMV scale table",
    );
    require(
        GEMM_Q4_MLX,
        "sb[(ulong)row * gpr + g]",
        "MLX Q4 GEMM scale table",
    );
    forbid(GEMV_Q4, "const uint row_base = row * cols;", "Q4 GEMV");
    forbid(
        GEMV_Q4_MLX,
        "const uint scale_base = row * groups_per_row;",
        "MLX Q4 GEMV",
    );
    forbid(
        GEMV_Q8,
        "const uint gi = row * groups_per_row + g;",
        "Q8 GEMV scale table",
    );
    forbid(GEMM_Q4_MLX, "sb[row * gpr + g]", "MLX Q4 GEMM scale table");
}

#[test]
fn embedding_table_offsets_are_widened_before_multiplication() {
    let token = 1u32 << 16;
    let hidden = 1u32 << 16;
    assert_eq!(token.wrapping_mul(hidden), 0);
    assert_eq!(u64::from(token) * u64::from(hidden), 1u64 << 32);

    require(
        EMBED,
        "const ulong total = (ulong)n_tokens * hidden;",
        "embed grid guard",
    );
    require(
        EMBED,
        "const ulong idx = (ulong)tid * hidden + d;",
        "embed packed row",
    );
    require(
        EMBED,
        "const ulong scale_i = (ulong)tid * groups_per_row + g;",
        "embed scale table",
    );
    require(
        DISPATCH_RS,
        "if n > u32::MAX as usize",
        "1D grid representability guard",
    );
    forbid(
        EMBED,
        "const uint total = n_tokens * hidden;",
        "embed grid guard",
    );
    forbid(
        EMBED,
        "const uint idx = tid * hidden + d;",
        "embed packed row",
    );
    forbid(
        EMBED,
        "const uint scale_i = tid * groups_per_row + g;",
        "embed scale table",
    );
}

#[test]
fn strided_shader_scans_cannot_roll_over_to_zero() {
    let step = 1_024u32;
    let last = u32::MAX - (u32::MAX % step);
    assert!(last < u32::MAX);
    assert_eq!(last.wrapping_add(step), 0);

    for (label, source, widened) in [
        (
            "row reductions",
            REDUCE,
            "for (ulong c = lid; c < (ulong)cols; c += tptg)",
        ),
        (
            "RMSNorm",
            RMS_NORM,
            "for (ulong d = lid; d < (ulong)dim; d += tptg)",
        ),
        (
            "one-pass argmax",
            SOFTCAP_SAMPLE,
            "for (ulong i = lid; i < (ulong)n; i += tptg)",
        ),
        (
            "MLX Q4 GEMV",
            GEMV_Q4_MLX,
            "for (ulong k0 = 0ul; k0 < (ulong)cols; k0 += SIMD_BLOCK)",
        ),
        (
            "MLX Q4 GEMM",
            GEMM_Q4_MLX,
            "for (ulong k0 = 0ul; k0 < (ulong)cols; k0 += SIMD_BLOCK)",
        ),
    ] {
        require(source, widened, label);
    }
    require(
        GEMV_Q4,
        "for (ulong g = tid; g < (ulong)groups_per_row; g += GEMV_TG)",
        "tiled Q4 group scan",
    );
    require(
        GEMV_Q8,
        "for (ulong i = lane; i < (ulong)group_size; i += Q8_SIMD_SIZE)",
        "Q8 scalar group scan",
    );
    for (label, source, narrow) in [
        (
            "row reductions",
            REDUCE,
            "for (uint c = lid; c < cols; c += tptg)",
        ),
        (
            "RMSNorm",
            RMS_NORM,
            "for (uint d = lid; d < dim; d += tptg)",
        ),
        (
            "one-pass argmax",
            SOFTCAP_SAMPLE,
            "for (uint i = lid; i < n; i += tptg)",
        ),
        (
            "MLX Q4 GEMV",
            GEMV_Q4_MLX,
            "for (uint k0 = 0u; k0 < cols; k0 += SIMD_BLOCK)",
        ),
        (
            "MLX Q4 GEMM",
            GEMM_Q4_MLX,
            "for (uint k0 = 0u; k0 < cols; k0 += SIMD_BLOCK)",
        ),
    ] {
        forbid(source, narrow, label);
    }
    forbid(
        GEMV_Q4,
        "for (uint g = tid; g < groups_per_row; g += GEMV_TG)",
        "tiled Q4 group scan",
    );
    forbid(
        GEMV_Q8,
        "for (uint i = lane; i < group_size; i += Q8_SIMD_SIZE)",
        "Q8 scalar group scan",
    );
}

/// The KV cache is where a `u32` product is actually reachable.
///
/// `q_head_base` is `b * Tq * H * D`. At a 8192-token context with 32 heads of
/// 128 that is 33.5M elements per batch element, so a batch of 128 crosses
/// `u32::MAX` — and a wrapped base does not fault, it reads another sequence's
/// keys and returns plausible attention. Every one of these kernels widens
/// before multiplying; this is what keeps it that way.
#[test]
fn attention_and_kv_cache_offsets_are_widened_before_multiplication() {
    let batch = 1u32 << 8;
    let per_batch = 1u32 << 24;
    assert_eq!(batch.wrapping_mul(per_batch), 0);
    assert_eq!(u64::from(batch) * u64::from(per_batch), 1u64 << 32);

    for (label, source) in [
        ("decode attention", ATTN_DECODE),
        ("rows attention", ATTN_ROWS),
        ("sliding-window h128", ATTN_SWA_128),
        ("sliding-window h256", ATTN_SWA_256),
        ("global h512", ATTN_GLOBAL),
    ] {
        require(source, "(ulong)b * kv_capacity * kv_pos_stride", label);
        forbid(source, "b * kv_capacity * kv_pos_stride + hkv", label);
    }
    // The decode kernel holds one query position per dispatch, so it has no
    // `Tq`-strided query base to widen; the other four do.
    for (label, source) in [
        ("rows attention", ATTN_ROWS),
        ("sliding-window h128", ATTN_SWA_128),
        ("sliding-window h256", ATTN_SWA_256),
        ("global h512", ATTN_GLOBAL),
    ] {
        require(source, "(ulong)b * Tq * q_pos_stride", label);
        forbid(source, "b * Tq * q_pos_stride + h *", label);
    }

    require(
        KV_STORE,
        "const ulong dst_offset = (ulong)*dst_offset_ptr;",
        "KV timestep store offset",
    );
    require(
        KV_STORE,
        "src[src_t * slot_width + e]",
        "KV ring densify source address",
    );
    // The bounds test is written as a subtraction precisely so that
    // `dst_offset + n` cannot wrap back inside the buffer.
    require(
        KV_STORE,
        "if (dst_offset > capacity || count > capacity - dst_offset) return;",
        "KV store bounds test",
    );
    forbid(
        KV_STORE,
        "if (dst_offset + count > capacity) return;",
        "KV store bounds test",
    );

    require(
        QKV_ROPE,
        "const ulong total_q = (ulong)T * (ulong)Hq;",
        "QKV RoPE query extent",
    );
    require(
        QKV_ROPE,
        "q + ((t * (ulong)Hq + h) * (ulong)D)",
        "QKV RoPE query row address",
    );
    forbid(
        QKV_ROPE,
        "const uint total_q = T * Hq;",
        "QKV RoPE query extent",
    );
}

/// No kernel may join the build without a decision about this suite.
///
/// These tests inspect a hand-listed set of sources, which is silent by
/// construction about any file nobody remembered to add: eleven of the crate's
/// twenty-one kernels — including all five attention kernels and the KV cache —
/// were unlisted, so "shader index arithmetic is pinned" was true of half the
/// shaders and unexamined for the rest. A file must now be on one list or the
/// other, and the exempt list carries the reason.
#[test]
fn every_kernel_source_is_inspected_or_explicitly_exempt() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels");
    let mut found = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(root)];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("kernels directory is readable") {
            let path = entry.expect("readable directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "metal") {
                let rel = path
                    .strip_prefix(root)
                    .expect("walked from the kernels root")
                    .to_string_lossy()
                    .into_owned();
                found.push(rel);
            }
        }
    }
    found.sort();
    assert!(!found.is_empty(), "no kernel sources found under {root}");

    let mut accounted: Vec<&str> = INSPECTED_KERNELS.to_vec();
    accounted.extend(EXEMPT_KERNELS.iter().map(|(name, _)| *name));
    accounted.sort_unstable();

    let unaccounted: Vec<&String> = found
        .iter()
        .filter(|name| !accounted.contains(&name.as_str()))
        .collect();
    assert!(
        unaccounted.is_empty(),
        "these kernel sources are neither inspected nor exempt: {unaccounted:?}. \
         Add index-arithmetic assertions for them, or list them in \
         EXEMPT_KERNELS with the reason they need none."
    );

    let stale: Vec<&&str> = accounted
        .iter()
        .filter(|name| !found.iter().any(|f| f == *name))
        .collect();
    assert!(
        stale.is_empty(),
        "these kernel sources are listed but no longer exist: {stale:?}"
    );

    for (name, reason) in EXEMPT_KERNELS {
        assert!(
            reason.len() > 40,
            "exemption for {name} needs a reason, not a placeholder"
        );
    }
}

#[test]
fn utility_and_simdgroup_pointer_math_is_explicitly_wide() {
    require(
        UTILS,
        "if ((ulong)gid >= (ulong)rows * cols) return;",
        "transpose extent guard",
    );
    require(
        SIMD_GEMM,
        "A + (ulong)row0 * K + k0",
        "simdgroup A tile address",
    );
    require(SIMD_GEMM, "C[(ulong)r*N+c]", "simdgroup edge store address");
    forbid(
        UTILS,
        "if (gid >= rows * cols) return;",
        "transpose extent guard",
    );
    forbid(SIMD_GEMM, "A + row0 * K + k0", "simdgroup A tile address");
    forbid(SIMD_GEMM, "C[r*N+c]", "simdgroup edge store address");

    // TensorOps still uses some `uint` tile arithmetic. That is safe only while
    // every public GEMM operand stays below the stricter signed-32-bit MPP
    // contract, which also puts every logical element address below `u32::MAX`.
    require(
        GEMM_RS,
        "if t.numel() > i32::MAX as usize",
        "TensorOps host addressability contract",
    );
    assert!((i32::MAX as u64) < u64::from(u32::MAX));
}
