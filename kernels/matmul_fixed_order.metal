// Fixed-order f32 GEMM. One output element per thread. K walks ascending
// from 0. Each step stores the product, then adds it into an accumulator
// that started at +0. That is two roundings and not an FMA. build.rs
// compiles this file with -ffp-contract=off, and the pragma below keeps a
// later edit from contracting the pair. The reduction is never split.
//
// f32 subnormal operands, products and partial sums are flushed to zero on
// the GPU, so the fold is bit-identical to an IEEE CPU fold only when none
// occurs: see the "# Subnormals" section of gemm_fixed_order_nn in src/gemm.rs.
//
// This is the Metal Numerics::Exact product. It is not matmul2d and not
// simdgroup_matrix: those tiles match this fold on short K and on integer
// products, and they differ on order-sensitive K that fills a hardware
// chunk (measured 8x16x16, 40 of 128 outputs).
//
// Addresses are ulong. The host refuses an operand over i32::MAX elements
// before encode (validate_gemm / require_i32_extent).

#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off) reassociate(off)

kernel void matmul_fixed_order_nn_f32(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint i = tid.y;
    uint j = tid.x;
    if (i >= M || j >= N) {
        return;
    }
    float acc = 0.0f;
    for (uint p = 0u; p < K; ++p) {
        float a = A[(ulong)i * (ulong)K + (ulong)p];
        float b = B[(ulong)p * (ulong)N + (ulong)j];
        float prod = a * b;
        acc = acc + prod;
    }
    C[(ulong)i * (ulong)N + (ulong)j] = acc;
}

kernel void matmul_fixed_order_nt_f32(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint i = tid.y;
    uint j = tid.x;
    if (i >= M || j >= N) {
        return;
    }
    // B is stored [N, K]. C[i, j] = sum_p A[i, p] * B[j, p].
    float acc = 0.0f;
    for (uint p = 0u; p < K; ++p) {
        float a = A[(ulong)i * (ulong)K + (ulong)p];
        float b = B[(ulong)j * (ulong)K + (ulong)p];
        float prod = a * b;
        acc = acc + prod;
    }
    C[(ulong)i * (ulong)N + (ulong)j] = acc;
}

kernel void matmul_fixed_order_tn_f32(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint i = tid.y;
    uint j = tid.x;
    if (i >= M || j >= N) {
        return;
    }
    // A is stored [K, M], B [K, N]. C[i, j] = sum_p A[p, i] * B[p, j].
    float acc = 0.0f;
    for (uint p = 0u; p < K; ++p) {
        float a = A[(ulong)p * (ulong)M + (ulong)i];
        float b = B[(ulong)p * (ulong)N + (ulong)j];
        float prod = a * b;
        acc = acc + prod;
    }
    C[(ulong)i * (ulong)N + (ulong)j] = acc;
}
