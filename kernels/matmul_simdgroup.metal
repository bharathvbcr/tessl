// Portable tiled GEMM via simdgroup_matrix — A/B baseline vs TensorOps.
// C = A @ B for row-major f32 matrices (A: MxK, B: KxN, C: MxN).
//
// Tile geometry: 2×2 simdgroups of 8×8 → 16×16 output per threadgroup.
// Aligned dispatch uses this fast path; arbitrary shapes use the edge kernel below.

#include <metal_stdlib>
using namespace metal;

kernel void matmul_simdgroup_f32(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint TM = 8;
    constexpr uint TN = 8;
    constexpr uint TK = 8;
    constexpr uint SG_M = 2;
    constexpr uint SG_N = 2;

    const uint sg_m = sid / SG_N;
    const uint sg_n = sid % SG_N;
    const uint row0 = (tgpig.y * SG_M + sg_m) * TM;
    const uint col0 = (tgpig.x * SG_N + sg_n) * TN;

    // Out-of-range tiles (partial grid) leave their region untouched.
    if (row0 >= M || col0 >= N) {
        return;
    }

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, TM, TN>(0.0f);

    for (uint k0 = 0; k0 < K; k0 += TK) {
        simdgroup_float8x8 a_tile;
        simdgroup_float8x8 b_tile;
        // Assumes K % 8 == 0 and tiles fully in-bounds (Phase 0 contract).
        simdgroup_load(a_tile, A + row0 * K + k0, K, ulong2(0, 0), false);
        simdgroup_load(b_tile, B + k0 * N + col0, N, ulong2(0, 0), false);
        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
    }

    simdgroup_store(acc, C + row0 * N + col0, N, ulong2(0, 0), false);
    (void)lane;
}

// Partial tiles use independent per-simdgroup scratch. Keep the aligned kernel
// above free of threadgroup-memory allocation and boundary predicates.
kernel void matmul_simdgroup_edges_f32(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row0 = group.y * 16 + (sid / 2) * 8;
    const uint col0 = group.x * 16 + (sid % 2) * 8;
    // Uniform for the whole simdgroup; only simdgroup barriers appear below.
    if (row0 >= M || col0 >= N) return;
    threadgroup float tile_a[4][64];
    threadgroup float tile_b[4][64];
    threadgroup float tile_c[4][64];
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < K; k0 += 8) {
        for (uint i = lane; i < 64; i += 32) {
            const uint r = i / 8, c = i % 8;
            tile_a[sid][i] = (row0+r < M && k0+c < K) ? A[(row0+r)*K+k0+c] : 0.0f;
            tile_b[sid][i] = (k0+r < K && col0+c < N) ? B[(k0+r)*N+col0+c] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 a, b;
        simdgroup_load(a, tile_a[sid], 8);
        simdgroup_load(b, tile_b[sid], 8);
        simdgroup_multiply_accumulate(acc, a, b, acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc, tile_c[sid], 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < 64; i += 32) {
        const uint r = row0 + i / 8, c = col0 + i % 8;
        if (r < M && c < N) C[r*N+c] = tile_c[sid][i];
    }
}
