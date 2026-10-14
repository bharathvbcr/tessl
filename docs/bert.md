# BERT and DistilBERT learned sparse document encoders

`tessl::bert` runs BERT-family masked language models (`BertForMaskedLM` and
`DistilBertForMaskedLM`, such as the
`opensearch-neural-sparse-encoding-doc-*` model family) as document-side learned
sparse encoders, completely on Metal 4 using tessl's native kernels.

Models are loaded directly from safetensors checkpoints (`model.safetensors` +
`config.json`).

## What it computes

Learned sparse retrieval represents unstructured text as high-dimensional,
sparse vocabulary vectors. For a sequence of token IDs, the encoder computes:

$$
\text{sparse}[v] = \max_{t} \log(1 + \text{relu}(\text{logits}[t, v]))
$$

where $\text{logits}[t, v]$ are the masked language model predictions at token
position $t$ for vocabulary entry $v$. A search engine (such as OpenSearch or
Lucene) indexes non-zero dimensions directly as inverted-index term weights.
Padding tokens take no part in the reduction, contributing zero to the final
document representation.

## Architecture

### Transformer Layer

Each layer computes:

1. **Embedding Stage:**
   - BERT: $\text{LayerNorm}(\text{word}[\text{ids}] + \text{position}[t] + \text{type}[0])$
   - DistilBERT: $\text{LayerNorm}(\text{word}[\text{ids}] + \text{position}[t])$
2. **Self-Attention:**
   - Projections: $Q = X W_q + b_q$, $K = X W_k + b_k$, $V = X W_v + b_v$
   - Bidirectional Attention: $\text{softmax}(Q K^T / \sqrt{d_k} + \text{mask}) V$ via `crate::embedgemma2::encoder_attn`
   - Residual & Post-Attention Norm: $\text{LayerNorm}(A W_o + b_o + X)$ (fused in `bert_layer_norm_residual_f32`)
3. **Feed-Forward Network (FFN):**
   - Exact GELU (erf): $\text{gelu}(X W_i + b_i)$ via `bert_bias_gelu_erf_f32`
   - Down-projection & Output Norm: $\text{LayerNorm}(\text{FFN}(X) W_{\text{out}} + b_{\text{out}} + X)$
4. **Prediction Head:**
   - Decoder transform: $\text{LayerNorm}(\text{gelu}(X W_t + b_t))$
   - Projection: $X_{\text{pred}} W_{\text{word}}^T + b_{\text{decoder}}$
   - Segment Max Pooling: vocabulary-wide sparse pooling accumulated across sequence rows via `bert_segment_sparse_max_f32`

## Metal Kernels (`kernels/bert.metal`)

| Kernel Entry Point | Operation |
|---|---|
| `bert_layer_norm_f32` | Standard LayerNorm across hidden dimension with affine parameters ($w, b$) and threadgroup reduction |
| `bert_layer_norm_residual_f32` | Fused residual addition + bias addition + LayerNorm: $Y = \text{LayerNorm}(X + R + b)$ in a single memory pass |
| `bert_embed_layer_norm_f32` | Fused word embedding lookup + position embedding + token type addition + LayerNorm |
| `bert_bias_add_f32` | In-place elementwise bias broadcast: $X_{r, c} += b_c$ |
| `bert_bias_gelu_erf_f32` | In-place exact erf GELU with bias addition: $X_{r, c} = \text{gelu}(X_{r, c} + b_c)$ matching PyTorch `torch.nn.functional.gelu(..., approximate='none')` |
| `bert_segment_sparse_max_f32` | Segmented vocabulary max-reduction and activation: $\text{pooled}[s, v] = \max(\text{pooled}[s, v], \log(1 + \text{relu}(\max_{r} \text{logits}[r, v] + b_v)))$ |

## SafeTensors Checkpoint Loading

`BertSparseModel::load(rt: &Arc<GpuRuntime>, st: &SafeTensors, cfg: BertConfig)`
takes an opened checkpoint and a parsed config; `BertConfig::from_config_file`
(or `from_config_json`) reads the model's `config.json`. Every tensor's shape is
checked against `cfg`, every weight must be finite, and a checkpoint tensor the
model never reads is an error. `encode(&self, batch: &[&[u32]], trace: bool)`
returns a `SparseOutput`:

```rust
use std::path::Path;
use tessl::bert::{BertConfig, BertSparseModel};
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

let rt = GpuRuntime::new()?;
let dir = Path::new("models/opensearch-neural-sparse-encoding-doc-v2-distilbert");
let cfg = BertConfig::from_config_file(&dir.join("config.json"))?;
let st = SafeTensors::open(&dir.join("model.safetensors"))?;
let model = BertSparseModel::load(&rt, &st, cfg)?;

// Encode a batch of tokenized sequences (the tokenizer's [CLS] and [SEP]
// already in the ids, each at most `max_positions` long):
let batch: Vec<Vec<u32>> = vec![
    vec![101, 2054, 2003, 1037, 13997, 102], // "[CLS] what is a document [SEP]"
];
let slices: Vec<&[u32]> = batch.iter().map(|s| s.as_slice()).collect();

// `trace = false`; `trace = true` takes exactly one sequence.
let out = model.encode(&slices, false)?;
// `out.pooled`: Vec<f32>, [batch, vocab], row b is sequence b's sparse vector.
// `out.forwards`: how many forwards the batch was split into.
```

There is no top-k extraction method: `encode` returns the full-vocabulary
vector, and selecting the non-zero `(term_id, weight)` pairs for an inverted
index is the caller's job. `SparseOutput::trace` is empty unless `trace` is set;
with it, it holds the embedding LayerNorm output and the residual stream after
each layer, `[tokens, hidden]` each.

## Batching & Memory Safety

- **`MAX_BATCH = 256`**: Maximum number of sequences accepted per `encode` call.
- **`MAX_BATCH_TOKENS = 8192`**: The encoder partitions batches into forwards of at most 8,192 padded tokens, bounding the intermediate FFN activation memory to 96 MiB.
- **`HEAD_BLOCK_ROWS = 2048`**: Logits ($30,522$ vocabulary entries) are evaluated and pooled in chunks of 2,048 rows (~250 MB), eliminating the need to materialize the full logits tensor for large batches.

## Numerical Parity & Verification

Every kernel in `kernels/bert.metal` is verified against high-precision float64 or PyTorch reference implementations in `tests/bert_kernels.rs`:
- LayerNorm (plain, fused and after the embedding sum): within `1e-5 * max(1, max|ref|)` of the f64 reference.
- erf GELU: within `(1e-6 + 4 x^2 eps) * |ref| + 1e-30`, a relative bound element by element, not bit-for-bit; the tanh approximation must fail the same bound.
- Segment Sparse Max: tested with randomized logits and variable segment lengths against reference CPU reduction.

The composed forward is held to transformers in `tests/bert_model.rs`: a tiny random `BertForMaskedLM` and `DistilBertForMaskedLM` (`tests/fixtures/bert_tiny/`) are encoded as one padded batch and compared with transformers' fp32 forward of each sequence alone (pooled sparse vectors, the embedding output and every layer's output; bound `2e-4 + 2e-4 * |want|`, stated in the test). The fixtures come from `scripts/gen_bert_fixtures.py` (run with the oracle venv), and `tests/fixtures/bert_tiny/manifest.json` records the torch, transformers and numpy versions, the seed, and the sha256 of the generator and of every file. Exchanging two layers' weights fails the test.
