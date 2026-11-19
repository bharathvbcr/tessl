# TimesFM 3

`tessl::timesfm3` runs Google Research's TimesFM 3 time-series foundation model
(`google/timesfm-3.0-pytorch`) on the Mac's GPU, from `config.json` and
`model.safetensors` to quantile forecasts, with no Python on the path. It ports
upstream's PyTorch implementation
([google-research/timesfm](https://github.com/google-research/timesfm) at
`918df02`, `src/timesfm3/torch/`, Apache-2.0) in three layers:

| Layer | Module | Upstream |
| :--- | :--- | :--- |
| Forecaster: ragged series, NaNs, options, grouping | `tessl::timesfm3_forecast`, `Timesfm3Model::predict_batch` | `TimesFM3Forecaster.predict_batch()` |
| Host steps around the model: padding, detrending, RevIN, CPM refinement, stitching | `tessl::timesfm3_host`, `Timesfm3Model::decode` | `TimesFM3Torch.decode()` |
| The model: residual block, 20-layer mixing transformer, output head (exact f32 GEMMs, `kernels/timesfm3.metal` for QK prep and head-dim-80 attention) | `Timesfm3Model::forward` | `TimesFM3Torch.forward()` |

The weights are under the **TimesFM Non-Commercial License v1.0**. tessl reads
them and never ships them, and nothing computed from them is committed: the
real-checkpoint references live under `target/`.

## Getting the checkpoint (no Python)

```sh
REV=43046b85ec22d584a13f8098c2ed39c889e129c2
DIR=~/.cache/huggingface/hub/models--google--timesfm-3.0-pytorch/snapshots/$REV
mkdir -p "$DIR"
for f in config.json model.safetensors LICENSE; do
  curl -fL -o "$DIR/$f" "https://huggingface.co/google/timesfm-3.0-pytorch/resolve/$REV/$f"
done
shasum -a 256 "$DIR/model.safetensors"
# a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da  (1,322,898,824 bytes, 330.7M F32 parameters)
```

The repository is not gated. Read `LICENSE` before using the weights.

## Command line

```sh
cargo build --release --bin timesfm3_forecast
target/release/timesfm3_forecast --model "$DIR" --horizon 96 --input series.csv > forecast.csv
```

Input is long-format CSV with a header naming `id` and `value` and one row
per point in time order. An empty value or `NaN` is a missing point. Output
is `id,step,point,q0.1,…,q0.9`, where `point` is the median quantile (upstream's
`forecast`). `--batch-size`, `--znorm`, `--symmetric`, `--positive` and
`--no-sort` are upstream's `per_core_batch_size`, `use_znorm`,
`use_symmetric_averaging`, `make_positive` and `sort_quantiles=False`. The
command line takes one target variate per series; the library takes several,
and covariates.

## Library

```rust,ignore
use tessl::safetensors::SafeTensors;
use tessl::timesfm3::{Timesfm3Config, Timesfm3Model};
use tessl::timesfm3_forecast::{PredictOptions, Rows, Series};

let rt = tessl::GpuRuntime::new()?;
let cfg = Timesfm3Config::from_config_file(&dir.join("config.json"))?;
let model = Timesfm3Model::load(&rt, &SafeTensors::open(&dir.join("model.safetensors"))?, cfg)?;
let series = vec![Series { target: Rows::one(history), ..Series::default() }];
let out = model.predict_batch(&series, 96, &PredictOptions::default())?;
let median: Vec<f32> = out[0].point(); // [variates, horizon]; out[0].values is [variates, horizon, quantiles]
```

`Series::past_only` and `Series::past_future` carry covariates (`[channels,
context]` and `[channels, context + horizon]`). `Timesfm3Model::decode` is
upstream's `decode()` on one rectangular batch, for callers that do their own
batching.

## Compatibility with upstream at 918df02

| Upstream | Status |
| :--- | :--- |
| `predict()` / `predict_batch()`: every option (`use_symmetric_averaging`, `make_positive`, `sort_quantiles`, `use_znorm`, `padding_mode` `none`/`edge`, `per_core_batch_size`, quantiles and point forecast) | Ported; held to upstream on the tiny and real checkpoints (below) |
| Ragged contexts, leading/interior/trailing NaNs, all-NaN series and variates, contexts past the 15360 global context | Ported, same tests |
| Several target variates, past-only and past+future covariates, mixed within a group | Ported, same tests |
| `decode()` with per-variate masks; its global `mask` | Ported; the global mask is folded into the per-variate masks, which is all upstream does with it |
| `from_pretrained(local dir)` | `Timesfm3Config::from_config_file` + `Timesfm3Model::load` |
| `from_pretrained(hub id)`, `cache_dir`, `token`, `revision` | Not ported: no network code. Fetch with curl (above) |
| `.pth` / `.pt` checkpoints | Not ported: safetensors only |
| `use_frozen_running_stats=True`, `input_transform` other than `identity`, non-RMS norms, `v_norm`, non-ReLU, biases, `use_rope_var`, non-causal attention, head dim other than 80 | Refused by name at `config.json` (`Timesfm3Config::from_config_json`); the published checkpoint uses none of them |
| `use_variate_attention`, `use_stitching`, `use_linear_detrending` (and threshold), `use_iterative_cpm_revin`, `value_clip`, `use_memory_efficient_attention` | Read from `config.json`, both settings computed |
| `return_aux_outputs` | Not ported; `Timesfm3Model::forward(.., trace = true)` returns every layer's residual stream instead |
| `TimesFM3Evaluator` (evaluation helper) | Not ported |
| Training / fine-tuning | Not in tessl: ojas owns training (README, ABI 14) |
| `device`, `use_sdpa` | Not applicable: Metal, one attention kernel |

## Limits

- **Patch rows per forward.** `MAX_ROWS` = 8192 rows of `b × v × n`.
  `predict_batch` runs a larger group as several forwards of whole series.
  That changes nothing, because series in a batch never interact: the tiny
  test holds the two to 1e-6 and measured them equal. One series is refused by name only when
  it alone exceeds the cap. At the full 15360 context that means more than 16
  variates, targets and covariates together.
- **Variates.** At most 32 per series (`MAX_VARIATES`, upstream's `max_variates`).
- **Precision.** f32 weights and activations with exact f32 GEMMs; the
  runtime's relaxed precision must be off.

## How it is checked

`tools/timesfm3_ref/make_reference.py` is the oracle. It runs upstream's own
code at the pinned commit, read by path from `target/` and never installed or
vendored. Python only generates files: no test or runtime path runs it. It
writes:

- **Tiny fixtures** (`tests/fixtures/timesfm3_tiny`, committed, with
  `MANIFEST.json`): a seeded 2-layer model with the real structure. They hold
  three `decode()` cases with every intermediate, plus float64 references.
  They also hold five forecaster cases, each recording every `decode()` call
  upstream's forecaster made and its final forecast.
- **Real references** (`target/timesfm3_ref_real`, never committed): five
  `decode()` cases, and six forecaster cases. The forecaster cases include
  contexts past 15360 and one group of about 15.5k patch rows.

| Test | What it holds | Measured on M5 Pro, 2026-11-19 (max \|diff\| / peak) |
| :--- | :--- | :--- |
| `timesfm3_host` (device-free) | Host steps against upstream's fp32 and f64 runs | — |
| `timesfm3_forecast_host` (device-free) | `plan` makes upstream's `decode()` calls; `finish` turns upstream's decode outputs into its forecast; malformed requests refused by name | values ≤ 3.3e-7 (z-norm rows; the rest exact), masks and shapes exact |
| `timesfm3_tiny` | Layer by layer, decode, `predict_batch` against upstream; a group in pieces equals one forward | forecaster ≤ 1.4e-5 (gate 5e-5); pieces 0 |
| `timesfm3_real` (`--ignored`) | Real checkpoint: layers 0/10/19, raw logits, decode, and `predict_batch` on all six forecaster cases | decode ≤ 7.6e-6, forecaster ≤ 7.0e-6 (gate 1e-3) |
| `timesfm3_forecast` binary, by hand | CSV in, CSV out on the real checkpoint (three series of 300, 90 and 700 points, a gap, `--positive`, horizon 96) against upstream's `predict_batch()` on the same CSV | ≤ 9.8e-7; same rows and quantile labels |

Mutation checks run against the host tests. Each of the following, applied
alone, turns `timesfm3_forecast_host` red:

- dropping the quantile reversal of symmetric averaging;
- not restoring the z-norm mean;
- not clamping in `make_positive`;
- never sorting quantiles;
- turning edge padding off;
- turning leading-NaN trimming off;
- holding interpolated values flat;
- removing the global-context cap;
- leaving placeholder covariates unmasked;
- fixing the group size at 4;
- not zeroing all-NaN series.

Disabling RoPE turns `timesfm3_tiny` red.

```sh
tools/timesfm3_ref/fetch_upstream.sh
~/.venvs/ml/bin/python -I tools/timesfm3_ref/make_reference.py --tiny --manifest --out tests/fixtures/timesfm3_tiny
~/.venvs/ml/bin/python -I tools/timesfm3_ref/make_reference.py --model-dir "$DIR" --out target/timesfm3_ref_real
TIMESFM3_DIR="$DIR" TIMESFM3_REF_DIR=target/timesfm3_ref_real \
  cargo test --release --test timesfm3_real -- --ignored --test-threads=1
```

The oracle environment is `tools/msl_emu/requirements.txt` plus `safetensors`.
Regenerating the tiny fixtures under torch 2.14.0 reproduced the 2.12.1 ones
byte for byte; only the version line in `case.json` changed.

## Not done yet

- **Timings.** `bench_timesfm3` (with `bench/timesfm3_torch.py` and
  `bench/timesfm3_mlx.py` lanes) measures end-to-end decode latency. No
  number is claimed until it runs under the machine's heavy-job lock on an
  idle GPU. The MLX lane also needs an MLX environment that does not exist on
  this machine yet.
- **bf16 operands, a C ABI and a Go binding** (as EmbeddingGemma 2 has). The
  last two change tessl's ABI and need the ojas handshake.
- **Training / fine-tuning** belongs to ojas.
