// Compile-only probe: int4b_format TensorOps operands. Pick a variant with
// -DV_<NAME>=1 (see run.sh). No kernel is executed.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// A is [M,K], B is [K,N], C is [M,N], row-major; extents/strides count
// elements (dim0 = fastest). Operands arrive as raw bytes: the tensor's
// data_handle_type is `device uchar *` for a numeric-format element
// (metal_tensor:253-263) and `device T *` otherwise.
#define TAIL(LT, RT, DT)                                                            \
kernel void k(device uchar *A [[buffer(0)]], device uchar *B [[buffer(1)]],         \
              device uchar *C [[buffer(2)]], constant int &M [[buffer(3)]],         \
              constant int &N [[buffer(4)]], constant int &K [[buffer(5)]]) {       \
    constexpr auto d = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false,    \
                                           false, false,                            \
                                           matmul2d_descriptor::mode::multiply);    \
    matmul2d<d, execution_simdgroups<4>> op;                                        \
    using TA = tensor<device LT, dextents<int32_t, 2>, tensor_inline>;              \
    using TB = tensor<device RT, dextents<int32_t, 2>, tensor_inline>;              \
    using TC = tensor<device DT, dextents<int32_t, 2>, tensor_inline>;              \
    TA tA((TA::data_handle_type)A, dextents<int32_t, 2>{K, M});                     \
    TB tB((TB::data_handle_type)B, dextents<int32_t, 2>{N, K});                     \
    TC tC((TC::data_handle_type)C, dextents<int32_t, 2>{N, M});                     \
    op.run(tA, tB, tC);                                                             \
}

#if defined(V_HALF_I4_F32)
TAIL(half, int4b_format, float)
#elif defined(V_HALF_U4_HALF)
TAIL(half, uint4b_format, half)
#elif defined(V_BF_I4_F32)
TAIL(bfloat, int4b_format, float)
#elif defined(V_BF_I4_BF)
TAIL(bfloat, int4b_format, bfloat)
#elif defined(V_I8_I4_I32)
TAIL(int8_t, int4b_format, int32_t)
#elif defined(V_U8_U4_I32)
TAIL(uint8_t, uint4b_format, int32_t)
#elif defined(V_LHS_I4)
TAIL(int4b_format, half, float)
#elif defined(V_BOTH_I4)
TAIL(int4b_format, int4b_format, int32_t)
#elif defined(V_HALF_U4_I32)
// Unlisted combination (MPPTensorOpsMatMul2d.h:52-61): negative control.
TAIL(half, uint4b_format, int32_t)
#elif defined(V_MLX_GROUP_LOOP)
// Design sketch for MLX Q4 (w = scale*q + bias, unsigned nibbles per
// the Q4MlxBank note in src/nn.rs, hence uint4b_format; group G along K): W is [N,K]
// nibbles, K contiguous (transpose_right), one matmul2d per group into a fresh
// cooperative destination, then scale/bias applied per element in registers.
// out[m,n] = sum_g ( s[n,g] * P_g[m,n] + b[n,g] * xsum[m,g] ).
constant constexpr int G = 32;
kernel void k(device half *A [[buffer(0)]], device uchar *W [[buffer(1)]],
              device float *C [[buffer(2)]], device const float2 *sb [[buffer(3)]],
              device const float *xsum [[buffer(4)]], constant int &M [[buffer(5)]],
              constant int &N [[buffer(6)]], constant int &K [[buffer(7)]],
              uint2 tg [[threadgroup_position_in_grid]]) {
    constexpr auto d = matmul2d_descriptor(32, 32, G, false, true, false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    using TW = tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    auto tA = tensor(A, dextents<int32_t, 2>{K, M}, array<int, 2>{1, K});
    TW tW((TW::data_handle_type)W, dextents<int32_t, 2>{K, N}, array<int, 2>{1, K});
    auto tC = tensor(C, dextents<int32_t, 2>{N, M}, array<int, 2>{1, N});
    const int tx = tg.x * 32, ty = tg.y * 32;
    auto sA0 = tA.slice(0, ty);
    auto sW0 = tW.slice(0, tx);
    auto acc = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sW0), float>();
    auto grp = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sW0), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc.set(i, 0.0f);
    const int groups = K / G;
    for (int g = 0; g < groups; ++g) {
        auto a = tA.slice(g * G, ty);
        auto w = tW.slice(g * G, tx);
        op.run(a, w, grp);
        for (uint16_t i = 0; i < grp.get_capacity(); ++i) {
            if (!grp.is_valid_element(i)) continue;
            const auto idx = grp.get_multidimensional_index(i);  // {n, m}
            const int n = tx + (int)idx[0], m = ty + (int)idx[1];
            const float2 p = sb[n * groups + g];
            acc[i] = fma(p.x, grp[i], fma(p.y, xsum[m * groups + g], acc[i]));
        }
    }
    acc.store(tC.slice(tx, ty));
}
#elif defined(V_TENSOR_ONLY_TYPED_PTR)
// Negative control: a typed `device int4b_format *` is not the handle type.
kernel void k(device int4b_format *B [[buffer(0)]], constant int &N [[buffer(1)]],
              constant int &K [[buffer(2)]]) {
    tensor<device int4b_format, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>{N, K});
    (void)tB;
}
#elif defined(V_STATIC_EXT_16)
// Static extent(0) not a multiple of 32 elements: rejected.
kernel void k(device uchar *B [[buffer(0)]]) {
    using TB = tensor<device int4b_format, extents<int32_t, 16, 64>, tensor_inline>;
    TB tB((TB::data_handle_type)B, extents<int32_t, 16, 64>{});
    (void)tB;
}
#endif
