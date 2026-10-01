// Holds the emulator's barriers to what ThreadSanitizer must see in them.
//
//   barrier_probe <mode>
//
// Each mode is one 64-thread threadgroup (two simdgroups) in which two threads
// write the same word. The `*_missing` modes drop the barrier between the two
// writes, so they are a race and TSan must report it; the others must run
// clean. build.sh runs every mode under MSL_EMU_SANITIZE=thread: a barrier TSan
// cannot see fails the clean modes, and one that synchronises more than a real
// barrier does passes the missing ones.
#include <metal_stdlib>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>

using namespace metal;
using metal::emu::Ids;
using metal::emu::launch;

// External, so the compiler cannot delete the stores being probed (in the
// anonymous namespace it did, and every mode passed with nothing to race on).
float word;

namespace {

void run(const char *mode) {
    const bool tg = std::strncmp(mode, "tg", 2) == 0;
    const bool missing = std::strstr(mode, "_missing") != nullptr;
    launch(uint3(1, 1, 1), uint3(64, 1, 1), 0, [&](const Ids &id, float *) {
        if (tg) {
            // Across simdgroups: lane 0 of simdgroup 1, then thread 0.
            if (id.lid == 32) word = 1;
            if (!missing) threadgroup_barrier(mem_flags::mem_threadgroup);
            if (id.lid == 0) word = 2;
        } else if (id.sg == 0) {
            // Within a simdgroup: lane 5, then lane 0.
            if (id.lane == 5) word = 1;
            if (!missing) simdgroup_barrier(mem_flags::mem_threadgroup);
            if (id.lane == 0) word = 2;
        }
    });
}

// Simdgroup 1 returns while simdgroup 0 keeps exchanging: the leaver's
// arrive_and_drop must neither deadlock the rest nor corrupt a later phase.
void drop() {
    float sums[3] = {};
    launch(uint3(1, 1, 1), uint3(64, 1, 1), 64, [&](const Ids &id, float *tgm) {
        tgm[id.lid] = float(id.lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (id.sg == 1) return;
        for (int r = 0; r < 3; ++r) {
            const float s = simd_sum(tgm[id.lane + 32] + float(r));
            if (id.lane == 0) sums[r] = s;
        }
    });
    // sum(32..63) + 32 r
    for (int r = 0; r < 3; ++r)
        if (sums[r] != 1520.0f + 32.0f * float(r)) {
            std::fprintf(stderr, "barrier_probe: drop: round %d summed %g\n", r, sums[r]);
            std::exit(1);
        }
}

// A kernel-scope threadgroup array as build.sh rewrites it: every thread finds
// its own slot unwritten (NaN) in every group of every launch, and after a
// barrier reads its neighbour's write. Fails without the poison on first
// registration (launch 1, group 1 sees zeros), without launch's poison before
// each group (group 2 sees group 1's writes), and without `static` (each
// thread's array is private, so the neighbour's slot stays NaN).
std::atomic<int> tg_static_bad{0};

void tg_static_kernel(uint lid) {
    static float a[64];
    metal::emu::tg_static(a);
    if (!std::isnan(a[lid])) tg_static_bad.fetch_add(1);
    a[lid] = float(lid + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (a[(lid + 1) % 64] != float((lid + 1) % 64 + 1)) tg_static_bad.fetch_add(1);
}

void tg_static_probe() {
    for (int l = 0; l < 2; ++l) {
        launch(uint3(2, 1, 1), uint3(64, 1, 1), 0, [&](const Ids &id, float *) { tg_static_kernel(id.lid); });
    }
    if (tg_static_bad.load() != 0) {
        std::fprintf(stderr, "barrier_probe: tg_static: %d bad reads of a shared threadgroup array\n",
                     tg_static_bad.load());
        std::exit(1);
    }
}

} // namespace

int main(int argc, char **argv) {
    const char *modes[] = {"tg", "tg_missing", "simd", "simd_missing", "drop", "tg_static"};
    if (argc != 2 || std::find_if(std::begin(modes), std::end(modes),
                                  [&](const char *m) { return std::strcmp(m, argv[1]) == 0; }) == std::end(modes)) {
        std::fprintf(stderr, "usage: barrier_probe tg|tg_missing|simd|simd_missing|drop|tg_static\n");
        return 2;
    }
    if (std::strcmp(argv[1], "drop") == 0) drop();
    else if (std::strcmp(argv[1], "tg_static") == 0) tg_static_probe();
    else run(argv[1]);
    return 0;
}
