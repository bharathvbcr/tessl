#ifndef TESSL_I8_TILE_WALK_H
#define TESSL_I8_TILE_WALK_H

// Copy of `tile_walk` from gemm_coop.h, with internal linkage.
//
// gemm_coop.h's `constexpr constant` symbols are program-scope. Including that
// header from a second .metal file duplicates them at metallib link. The walk
// itself must stay the one production GEMM uses (Morton on a square
// power-of-two grid, otherwise column panels once B has at least 2^23
// elements, otherwise row-major). A change here that disagrees with
// gemm_coop.h mis-places tiles on the int8 path only.

#include <metal_stdlib>
using namespace metal;

static inline uint2 i8_morton_decode_2d(uint c) {
    uint x = 0, y = 0;
#pragma unroll
    for (uint i = 0; i < 16; ++i) {
        x |= ((c >> (2 * i)) & 1u) << i;
        y |= ((c >> (2 * i + 1)) & 1u) << i;
    }
    return uint2(x, y);
}

static inline bool i8_tile_grid_is_morton(uint tiles_n, uint tiles_m) {
    return tiles_n == tiles_m && tiles_n != 0u && (tiles_n & (tiles_n - 1u)) == 0u;
}

static inline uint2 i8_tile_from_linear(uint linear, uint tiles_n, uint tiles_m) {
    if (i8_tile_grid_is_morton(tiles_n, tiles_m)) {
        return i8_morton_decode_2d(linear);
    }
    return uint2(linear % tiles_n, linear / tiles_n);
}

static inline uint2 i8_tile_from_linear_panel(uint linear, uint tiles_n, uint tiles_m, uint ph) {
    if (linear >= tiles_n * tiles_m) return uint2(tiles_n, tiles_m);
    uint band = linear / (ph * tiles_n);
    uint rem = linear - band * ph * tiles_n;
    uint local_h = min(ph, tiles_m - band * ph);
    return uint2(rem / local_h, band * ph + rem % local_h);
}

static constant ulong I8_PANEL_MIN_B_ELEMS = 1ul << 23;
static constant uint I8_PANEL_BAND_ROWS = 512u;

template <int SM>
static inline uint2 i8_tile_walk(uint linear, uint tiles_n, uint tiles_m, uint N, uint K) {
    if ((ulong)N * (ulong)K >= I8_PANEL_MIN_B_ELEMS && !i8_tile_grid_is_morton(tiles_n, tiles_m)) {
        return i8_tile_from_linear_panel(linear, tiles_n, tiles_m, I8_PANEL_BAND_ROWS / (uint)SM);
    }
    return i8_tile_from_linear(linear, tiles_n, tiles_m);
}

#endif
