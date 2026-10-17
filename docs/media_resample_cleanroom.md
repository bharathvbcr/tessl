# `media::resample`: a clean-room record

`src/media/resample.rs` reproduces what the `ffmpeg` binary's default resampler and
channel downmix do (`ffmpeg -i in.wav -ac 1 -ar R -f f32le out.raw`), bit for bit on
AArch64. The module was rewritten without access to FFmpeg's source code or to the
previous implementation. This file records what was consulted, what was run, and how
each rule in the module was derived. The working log (every command) was kept in
`target/cleanroom_log.md` and is not committed; this is its summary.

## What was consulted

* The task statement and the interface it gives (signatures only).
* `src/media/mod.rs` for the `MediaError` type; `src/media/audio.rs` for the *code*
  that calls the resampler (`SwrMix::new`, `mix_interleaved`, `SwrConverter::new`,
  `push`, `finish`) and `src/bin/bench_media_resample.rs` for its use of `resample`.
* The existing acceptance tests `tests/media_resample.rs` and the fixtures under
  `tests/fixtures/media_resample/`, and `scripts/gen_resample_fixtures.py`, which shows
  how those goldens were made with the `ffmpeg` binary. Their comments state a few
  structural facts (the tap-count formula, the phase cap, a mirrored end). Each was
  treated as a hypothesis and re-derived by experiment below. No test or script contains a
  table of filter coefficients or other constants that look copied from FFmpeg internals.
* FFmpeg's published user documentation: `ffmpeg -h full` (the *SWResampler AVOptions*:
  `filter_size` 32, `phase_shift` 10, `linear_interp` true, `exact_rational` true,
  `filter_type` kaiser, `kaiser_beta` 9, `clev`/`slev` 0.707107, `lfe_mix_level` 0,
  `rematrix_volume` 1) and `man ffmpeg-resampler` (`cutoff` default 0.97 for the native
  resampler; with `exact_rational` the phase count falls back to `1 << phase_shift`).
* Repository infrastructure that has nothing to do with the resampler: the fixture-manifest
  test and writer, the hosted-test script and manifest, the CI workflow's doc step, and the
  bench driver and its input generator.
* Published DSP facts: the Kaiser window, the modified Bessel function `I0` and its
  power series, windowed-sinc interpolation.
* Black-box runs of `/opt/homebrew/bin/ffmpeg` (version 9.0.2) on inputs generated
  for the purpose.

Not consulted, by construction: the previous `src/media/resample.rs` (removed with
`git rm` without being opened, and not shown by any `git show`/`log -p`/`diff`/`blame`),
any FFmpeg source, `tools/embedgemma2_ref/audio_loader_gap.py`, the `## Media decoding`
section of `docs/embedgemma2.md`, and the doc comments of `src/media/audio.rs` that
describe the resampler.

### One inadvertent exposure

My first search for callers (`rg "resample|Swr" src`) printed seven lines of the module doc
comment at the top of `src/media/audio.rs` (lines 2, 16, 17, 26, 27, 28 and 30). They were doc
comments about the audio path rather than the resampler's internals, but they said that
the default downmix is "each channel resampled, then mixed with libswresample's matrix"
and that the mean downmix mixes first and resamples one channel. The same statements are
in the task text and in `tests/media_resample.rs`. They were not relied on: the order of
resampling and mixing was decided by experiment (below, E9), and the float rounding of the
mix itself, which those lines do not mention, by E17.

## Experiments

All inputs are WAV files written by the experiment (float, or 16-bit where noted); all
goldens are the raw `f32` output of `ffmpeg -ac 1 -ar R`. Comparisons are on bits.

| # | Experiment | Result | Decision |
|---|---|---|---|
| E1 | Unit impulse at three neighbouring positions of a long silent clip, 48 kHz to 16 kHz; read the filter taps off the output | 100 taps, symmetric about tap 49 (`taps/2 - 1`) apart from the last tap, sum 1 | Window `taps/2 - 1` samples before the output position, `taps/2` after |
| E2 | Fit the taps to a Kaiser(beta 9) windowed sinc | half-width `taps/2`, cutoff `0.97 * out / in`, rounded to `f32`: all 100 taps equal bit for bit | The design formula; Bessel `I0` by its series |
| E3 | Impulses at positions 0 to 100 | an impulse at `k` also appears at `-k`: the start is mirrored about sample 0 *without* repeating it | `x[-i] = x[i]` |
| E4 | Impulses and random data near the end | the end is mirrored about the half-sample after the last sample (the last sample is repeated) | `x[n-1+i] = x[n-i]` |
| E5 | Output length for n = 0 to 519 and 473 longer lengths, 12 rate pairs | 0 for short clips, then a clean rule (below) with 0 mismatches over all of them | the start gate, the tail pad `ceil(B/2)` |
| E6 | Impulse trains (every output sees one impulse), 44.1 kHz to 16 kHz (160 phases), 22.05 to 16 (320), 11.025 to 16 (640), 8 to 16 (2), 16 to 48 (3), 44.1 to 48 (160), 32, 48, 96 kHz to 16 | every `(phase, tap)` coefficient equals `f32(design)` when the scale is `1 / sum(phase 0)` for *all* phases | one global scale from phase 0 |
| E7 | The same for up-conversion | cutoff 1 (not 0.97), and the phase-0 taps at nonzero whole distances are exactly 0, not the `1e-17` residue of `sin(pi k)` | `cutoff == 1` and whole `d` gives 0 |
| E8 | 97 kHz to 50 kHz (cutoff exactly 0.5, phase 0 taps at whole `cutoff * d`) | the `sin(pi k)` residues are present (no rule keyed on whole products) | the zero rule is for `cutoff == 1` only |
| E9 | Stereo 48 kHz to 16 kHz, all candidate orders | resample each channel, then mix gives 0 mismatches; mixing first gives about 80 percent | resample, then mix |
| E10 | Two-tap probes (random values in two taps of one window), classified by tap offset | taps whose indices differ by a multiple of 4 are combined by a fused multiply-add, all others by a plain add | four lanes, lane = `j mod 4` |
| E11 | Three- and four-tap probes in distinct lanes | `(l0 + l1) + (l2 + l3)` for taps below 96; different once taps of the last four are present | lane reduction order, and a separate tail |
| E12 | Whole-window search over where the lane loop stops, 100, 198, 66, 92, 46, 32 taps | stops at `8 * floor(taps / 8)`; the rest are fused one at a time onto the reduced sum; 0 mismatches | the dot product |
| E13 | 44101 and 12345 Hz to 16 kHz (more than 1024 phases) | 1024 phases; linear blend of two rows; `f = (rem * 1024 mod out) / out`; the row after 1023 is row 0 moved one tap along with a 0 entering | interpolating bank |
| E14 | The summation in that mode, outputs with `f = 0` | a plain sequential fused chain, no lanes | `dot_chain` |
| E15 | The blend arithmetic: candidate formulas against 104,074 single-tap outputs and 3,000 random-data outputs | `f32(v0 + f * f32(v1 - v0))` with `f` an exact `f64` fraction, one rounding, 0 mismatches (an `f32` `f` gives 59) | the blend |
| E16 | Unit impulse per channel at equal rates, 1 to 8 channels | weights `s s` (2), `s s 0` (3), `s s 1 .5` (4), `s s 1 .5 .5` (5), `s s 1 0 .5 .5` (6), plus one more `.5` each for 7 and 8; `s = sqrt(1/2)` as `f32` | `SwrMix` |
| E17 | Order of the channel sum, all permutations of the weighted channels, random data | exactly two weighted channels: `fma(w0, x0, w1 * x1)`; more: an ascending fused chain from the first product | `mix_frame` |
| E18 | 16-bit and float WAV with the same values; equal rates; 1, 2, 3, 6, 8 channels | identical output | no separate integer path |
| E19 | NaN, infinities, negative zero in a channel the mix gives no weight, in mono, and cancelling infinities | all equal to this module's output | zero-weight channels do not take part |

### The output-length rule (E5)

With `taps = T`, the clip of `n` samples, output `m` at input position `idx0(m) = floor(m * in / out)`:

* a window is the `T` samples from `idx0 - (T/2 - 1)` to `idx0 + T/2`;
* once `n > T`, every output whose window is entirely in the clip is produced as the
  samples arrive and the samples before the next window are dropped; `consumed` is the
  index of the first one kept (0 if `n <= T`);
* at the end, with `B = n - consumed` samples held, the clip is extended by `ceil(B / 2)`
  samples mirrored about its end and every output whose window fits in the extended clip
  is produced, but *only if* the extended clip is longer than `T`.

This gives 0 outputs for 61 samples and 18 for 62 at 44.1 to 16 kHz (92 taps), 34 for
exactly 100 samples at 48 to 16 kHz (the single case where dropping consumed samples
early would have changed the answer), and the right count for every one of the 12 rate
pairs over 520 lengths and 473 long ones.

## How the rules in the module were derived

* **Taps and phases.** `ceil(32 / min(1, 0.97 out / in))` rounded up to even: it matches the impulse
  support for every rate pair above (E1, E6); phases are `out / gcd(in, out)` while that is at most
  1024 (E6), else 1024 with interpolation (E13).
* **Coefficients.** `d = (j - (taps/2 - 1)) - p / phases`, sinc `sin(pi c d) / (pi d)` (`c` for `d = 0`),
  times `I0(9 sqrt(1 - (d / (taps/2))^2)) / I0(9)`, divided by the sum of phase 0, rounded to `f32` (E1, E2, E6, E7).
* **Positions.** Exact integer arithmetic: the product `m * in / out` and its remainder; the remainder
  is the phase (E6) or the blend fraction (E13).
* **Edges.** E3, E4, E5.
* **Sum order.** E10 to E12 for the exact banks, E14 and E15 for the interpolating ones, E17 for the mix.
* **Bounds** (`MAX_RATE`, `MAX_TAPS`, `MAX_BANK_FLOATS`, `MAX_SWR_CHANNELS`) are tessl's own limits and
  say nothing about FFmpeg.

## Parity achieved

| Check | Result |
|---|---|
| `tests/media_resample.rs` (44 clips: 10 rates to 16 kHz, 1 to 8 channels, impulses, steps, sweeps, noise, lengths 1 to 144,000) | passes unchanged; 0 samples differ, **max 0 ULP** |
| `tests/media_resample_cr.rs`: 105 clips including 23 rate pairs, all 8 layouts at 5 rates, impulse trains over the banks of 9 pairs, subnormals, NaN, infinity, negative zero | **0 samples differ, max 0 ULP** |
| 2,003 clip lengths over 12 rate pairs (including the equal-rate pair) | the output length and every sample equal `ffmpeg`'s |
| Arbitrary push chunking | output identical to the single push |

It is bit-exact for this FFmpeg build on AArch64. The fused multiply-add structure is what
the AArch64 build does; an x86 build of FFmpeg is not expected to match, and was not
available to test.

### Not verified

* Other FFmpeg versions or builds (only 9.0.2 was available).
* Clips longer than a few minutes, or sample rates outside the pairs listed.
* Whether the zero-weight channel and the stereo special order hold for layouts not in the default
  list (the module refuses those: more than 8 channels).
