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

`BertSparseModel::load` loads standard HuggingFace / OpenSearch models from a directory containing `model.safetensors` and `config.json`:

```rust
use std::path::Path;
use std::sync::Arc;
use tessl::bert::BertSparseModel;
use tessl::GpuRuntime;

let rt = Arc::new(GpuRuntime::new()?);
let model = BertSparseModel::load(rt, Path::new("models/opensearch-neural-sparse-encoding-doc-v2-distilbert"))?;

// Encode a batch of tokenized sequences:
let batch: Vec<Vec<u32>> = vec![
    vec![101, 2054, 2003, 1037, 13997, 102], // "[CLS] what is a document [SEP]"
];
let slices: Vec<&[u32]> = batch.iter().map(|s| s.as_slice()).collect();

// Full vocabulary sparse vector (f32 tensor [batch_size, vocab_size]):
let sparse_tensor = model.encode(&slices)?;

// Or directly extract top-k non-zero (term_id, weight) pairs for inverted index posting:
let top_k_postings = model.encode_topk(&slices, 64)?;
```

## Batching & Memory Safety

- **`MAX_BATCH = 256`**: Maximum number of sequences accepted per `encode` call.
- **`MAX_BATCH_TOKENS = 8192`**: The encoder partitions batches into forwards of at most 8,192 padded tokens, bounding the intermediate FFN activation memory to 96 MiB.
- **`HEAD_BLOCK_ROWS = 2048`**: Logits ($30,522$ vocabulary entries) are evaluated and pooled in chunks of 2,048 rows (~250 MB), eliminating the need to materialize the full logits tensor for large batches.

## Numerical Parity & Verification

Every kernel in `kernels/bert.metal` is verified against high-precision float64 or PyTorch reference implementations in `tests/bert_kernels.rs`:
- Fused LayerNorm: verified against f64 reference with epsilon bounds $\le 10^{-6}$.
- erf GELU: bit-level parity against exact mathematical erf formulation.
- Segment Sparse Max: tested with randomized logits and variable segment lengths against reference CPU reduction.
