// Measurement-only geometries of the encoder attention's matrix-unit kernels,
// linked only by `TESSL_GEMM_TUNE=1` builds and timed against production by
// `bench_encoder_attn`. Same entry points as production
// (`kernels/encoder_attn_tiled.h`), so a swept kernel is the shipped kernel at
// another BQ x BK x NSG, not a copy of it. Names are
// `encoder_attn_tune[_bf16|_bf16p32]_h{D}_q{BQ}_k{BK}_sg{NSG}`.
//
// A geometry must satisfy `attn_tiled_core`'s static_asserts: NSG*32 threads a
// multiple of BQ, at most 32 threads per query row, and BK a multiple of the
// threads per row. The S tile (BQ*BK f32) stays within 16 KiB so that, with
// the bf16 P tile and the row state, a threadgroup fits the 32 KiB limit.
#include "../encoder_attn_tiled.h"

#define TUNE_F32(D, BQ, BK, NSG) \
    ENCODER_TILED_KERNEL(encoder_attn_tune_h##D##_q##BQ##_k##BK##_sg##NSG, D, BQ, BK, NSG)
#define TUNE_BF16(D, BQ, BK, NSG) \
    ENCODER_TILED_BF16_KERNEL(encoder_attn_tune_bf16_h##D##_q##BQ##_k##BK##_sg##NSG, D, BQ, BK, NSG)
#define TUNE_BF16_PF32(D, BQ, BK, NSG) \
    ENCODER_TILED_BF16_PF32_KERNEL(encoder_attn_tune_bf16p32_h##D##_q##BQ##_k##BK##_sg##NSG, D, BQ, BK, NSG)

// Head dim 64 (the vision tower's), every valid geometry up to a 16 KiB S tile.
TUNE_F32(64, 32, 32, 1)
TUNE_F32(64, 32, 32, 2)
TUNE_F32(64, 32, 32, 4)
TUNE_F32(64, 32, 32, 8)
TUNE_F32(64, 32, 64, 1)
TUNE_F32(64, 32, 64, 2)
TUNE_F32(64, 32, 64, 4)
TUNE_F32(64, 32, 64, 8)
TUNE_F32(64, 32, 128, 2)
TUNE_F32(64, 32, 128, 4)
TUNE_F32(64, 32, 128, 8)
TUNE_F32(64, 64, 32, 2)
TUNE_F32(64, 64, 32, 4)
TUNE_F32(64, 64, 32, 8)
TUNE_F32(64, 64, 64, 2)
TUNE_F32(64, 64, 64, 4)
TUNE_F32(64, 64, 64, 8)
TUNE_F32(64, 128, 32, 4)
TUNE_F32(64, 128, 32, 8)

// bf16 operands, head dim 64: P rounded to bf16 (both operands bf16) and P
// kept f32 (mixed), at the geometries the f32 sweep makes plausible.
TUNE_BF16(64, 32, 32, 4)
TUNE_BF16(64, 64, 32, 4)
TUNE_BF16(64, 64, 64, 4)
TUNE_BF16(64, 64, 64, 8)
TUNE_BF16(64, 128, 32, 8)
TUNE_BF16_PF32(64, 32, 32, 4)
TUNE_BF16_PF32(64, 64, 32, 4)
TUNE_BF16_PF32(64, 64, 64, 4)

// bf16 operands at the text encoder's head dims: the f32 production geometry
// and its neighbours.
TUNE_BF16(256, 32, 32, 4)
TUNE_BF16(256, 32, 64, 4)
TUNE_BF16(256, 64, 32, 8)
TUNE_BF16(256, 64, 64, 8)
TUNE_BF16(512, 32, 32, 8)
TUNE_BF16(512, 32, 64, 8)
TUNE_BF16(512, 64, 32, 8)
TUNE_BF16_PF32(256, 32, 32, 4)
TUNE_BF16_PF32(512, 32, 32, 8)

// Second round of the bf16 sweep: wider and narrower neighbours of the first
// round's winners.
TUNE_BF16(64, 32, 64, 8)
TUNE_BF16(64, 32, 128, 4)
TUNE_BF16(64, 32, 128, 8)
TUNE_BF16(64, 64, 32, 8)
TUNE_BF16(64, 64, 64, 2)
TUNE_BF16(64, 64, 64, 16)
TUNE_BF16(64, 32, 32, 2)
TUNE_BF16(256, 32, 128, 4)
TUNE_BF16(256, 32, 64, 2)
TUNE_BF16(256, 32, 64, 8)
TUNE_BF16(256, 64, 64, 4)
TUNE_BF16(256, 16, 64, 4)
TUNE_BF16(256, 16, 64, 2)
TUNE_BF16(512, 32, 128, 8)
TUNE_BF16(512, 32, 64, 16)
TUNE_BF16(512, 32, 128, 16)
TUNE_BF16(512, 16, 64, 8)
TUNE_BF16(512, 16, 128, 8)
TUNE_BF16(512, 64, 64, 16)

// Third round: the second round's D=64 winner was the smallest threadgroup
// (two simdgroups), so go smaller and wider along that axis.
TUNE_BF16(64, 32, 32, 1)
TUNE_BF16(64, 32, 64, 1)
TUNE_BF16(64, 32, 64, 2)
TUNE_BF16(64, 32, 128, 2)
TUNE_BF16(64, 16, 32, 1)
TUNE_BF16(64, 16, 32, 2)
TUNE_BF16(64, 16, 64, 1)
TUNE_BF16(64, 16, 64, 2)
TUNE_BF16(64, 64, 32, 2)
TUNE_BF16(256, 32, 32, 2)
TUNE_BF16(256, 16, 32, 2)
TUNE_BF16(256, 16, 32, 4)
TUNE_BF16(512, 16, 32, 4)
TUNE_BF16(512, 16, 64, 4)
TUNE_F32(64, 16, 32, 1)
TUNE_F32(64, 16, 32, 2)
TUNE_F32(64, 16, 64, 2)

// Head dim 32 (BERT-small): the same template, to see whether it costs nothing.
TUNE_F32(32, 32, 32, 4)
TUNE_F32(32, 64, 32, 4)
TUNE_F32(32, 64, 64, 4)
TUNE_F32(32, 64, 64, 8)
TUNE_F32(32, 128, 32, 8)
TUNE_F32(32, 32, 32, 2)
TUNE_F32(32, 32, 64, 2)
TUNE_F32(32, 32, 32, 1)
TUNE_F32(32, 16, 32, 1)
