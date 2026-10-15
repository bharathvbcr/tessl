//! f64 references for EmbeddingGemma 2's vision tower: the axial RoPE, the
//! pooler written the way transformers writes it (a dense one-hot weight
//! matrix, independent of `plan_pooling`'s cell lists) and a whole host forward
//! composed from them and from `common::embedgemma2`'s text-encoder pieces.
//! `tests/embedgemma2_vision_model.rs` holds the forward to transformers by
//! golden; `tests/embedgemma2_vision_tiny.rs` runs it against the device.

use super::embedgemma2::{attn_ref, gelu_tanh, lin, rms, rope, Attn};

/// The shapes a reference forward needs (the model's `VisionConfig`, as plain
/// numbers so the reference does not depend on the code under test).
#[derive(Clone, Copy, Debug)]
pub struct VisionDims {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub patch_dim: usize,
    pub pool_k: usize,
    pub table_len: usize,
    pub theta: f64,
    pub text_hidden: usize,
}

/// `Gemma4VisionRotaryEmbedding` + `apply_multidimensional_rope` on one head
/// row: the first half of the channels rotates by `px` and the second by `py`,
/// each as `rotate_half` over its own `D / 2` channels, `inv_freq =
/// theta^(-2p / (D/2))`.
pub fn rope_axial(x: &[f64], px: f64, py: f64, theta: f64) -> Vec<f64> {
    let w = x.len() / 2;
    let mut out = rope(&x[..w], px, theta);
    out.extend(rope(&x[w..], py, theta));
    out
}

/// `Gemma4VisionPooler._avg_pool_by_positions` on `hidden: [P, h]` (padding
/// rows must be zero, as the pooler masks them), `positions: [P, 2]`: the
/// dense weight matrix `one_hot(cell) / k²`, `weights^T @ hidden`, and the
/// rows whose column of weights has any non-zero. Returns the kept rows
/// `[S, h]` in cell order and their cell ids. Panics where transformers
/// raises (a cell at or past `P / k²`).
pub fn pool_dense(hidden: &[f64], positions: &[i32], k: usize, h: usize) -> (Vec<f64>, Vec<usize>) {
    let total = positions.len() / 2;
    let length = total / (k * k);
    assert_eq!(length * k * k, total, "pool: k² times the output length must be P");
    let clamped: Vec<(i64, i64)> = positions
        .chunks_exact(2)
        .map(|p| (i64::from(p[0]).max(0), i64::from(p[1]).max(0)))
        .collect();
    let max_x = clamped.iter().map(|c| c.0).max().unwrap() + 1;
    let k_i = k as i64;
    let mut w = vec![vec![0.0f64; total]; length];
    for (p, &(x, y)) in clamped.iter().enumerate() {
        let cell = x / k_i + (max_x / k_i) * (y / k_i);
        assert!(
            (cell as usize) < length,
            "pool: patch {p} falls in cell {cell} of {length}"
        );
        w[cell as usize][p] = 1.0 / (k * k) as f64;
    }
    let (mut rows, mut cells) = (Vec::new(), Vec::new());
    for (c, wc) in w.iter().enumerate() {
        if wc.iter().all(|&v| v == 0.0) {
            continue;
        }
        cells.push(c);
        for j in 0..h {
            rows.push((0..total).map(|p| wc[p] * hidden[p * h + j]).sum());
        }
    }
    (rows, cells)
}

/// What [`vision_forward`] returns, for one image.
pub struct VisionRef {
    /// Real patches.
    pub n: usize,
    /// The patch embedder's output, `[n, hidden]`.
    pub patch_embed: Vec<f64>,
    /// The residual stream after each layer, `[n, hidden]`.
    pub layers: Vec<Vec<f64>>,
    /// After the pool and `* sqrt(hidden)`, `[S, hidden]`.
    pub pooled: Vec<f64>,
    /// After the embedder, `[S, text_hidden]`.
    pub soft: Vec<f64>,
    pub tokens: usize,
}

/// The vision tower and embedder of one image in f64, transformers' order.
/// `w(name)` returns a tensor by its full checkpoint name. The pixels are
/// scaled in f32 (`2 * (x - 0.5)`) as the host does; positions with `(-1, -1)`
/// are padding, a suffix.
pub fn vision_forward<'a>(
    d: &VisionDims,
    w: &dyn Fn(&str) -> &'a [f64],
    pixels: &[f32],
    positions: &[i32],
) -> VisionRef {
    let (h, hd, heads, inter) = (d.hidden, d.head_dim, d.heads, d.inter);
    let total = positions.len() / 2;
    let n = (0..total)
        .position(|t| positions[2 * t] == -1 && positions[2 * t + 1] == -1)
        .unwrap_or(total);
    let tower = |s: &str| w(&format!("vision_tower.{s}"));

    // Patch embedder.
    let (proj_w, table) = (
        tower("patch_embedder.input_proj.weight"),
        tower("patch_embedder.position_embedding_table"),
    );
    let mut resid: Vec<Vec<f64>> = (0..n)
        .map(|t| {
            let x: Vec<f64> = pixels[t * d.patch_dim..(t + 1) * d.patch_dim]
                .iter()
                .map(|&v| f64::from(2.0f32 * (v - 0.5f32)))
                .collect();
            let (px, py) = (positions[2 * t] as usize, positions[2 * t + 1] as usize);
            lin(proj_w, &x, h, d.patch_dim)
                .iter()
                .enumerate()
                .map(|(j, p)| p + (table[px * h + j] + table[(d.table_len + py) * h + j]))
                .collect()
        })
        .collect();
    let patch_embed = resid.concat();

    let mut layers = Vec::new();
    for li in 0..d.layers {
        let g = |s: &str| tower(&format!("encoder.layers.{li}.{s}"));
        let (mut q, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for (t, r) in resid.iter().enumerate() {
            let x = rms(r, Some(g("input_layernorm.weight")));
            let qp = lin(g("self_attn.q_proj.linear.weight"), &x, heads * hd, h);
            let kp = lin(g("self_attn.k_proj.linear.weight"), &x, heads * hd, h);
            let vp = lin(g("self_attn.v_proj.linear.weight"), &x, heads * hd, h);
            let (px, py) = (f64::from(positions[2 * t]), f64::from(positions[2 * t + 1]));
            for head in 0..heads {
                let s = head * hd..(head + 1) * hd;
                q.extend(rope_axial(
                    &rms(&qp[s.clone()], Some(g("self_attn.q_norm.weight"))),
                    px,
                    py,
                    d.theta,
                ));
                k.extend(rope_axial(
                    &rms(&kp[s.clone()], Some(g("self_attn.k_norm.weight"))),
                    px,
                    py,
                    d.theta,
                ));
                v.extend(rms(&vp[s], None));
            }
        }
        let a = Attn {
            b: 1,
            t: n,
            h: heads,
            hkv: heads,
            d: hd,
            window: 0,
            lens: vec![n],
        };
        let o = attn_ref(&a, &q, &k, &v, None);
        for (t, r) in resid.iter_mut().enumerate() {
            let y = lin(
                g("self_attn.o_proj.linear.weight"),
                &o[t * heads * hd..(t + 1) * heads * hd],
                h,
                heads * hd,
            );
            let y = rms(&y, Some(g("post_attention_layernorm.weight")));
            r.iter_mut().zip(&y).for_each(|(a, b)| *a += b);

            let x = rms(r, Some(g("pre_feedforward_layernorm.weight")));
            let gate = lin(g("mlp.gate_proj.linear.weight"), &x, inter, h);
            let up = lin(g("mlp.up_proj.linear.weight"), &x, inter, h);
            let mid: Vec<f64> = gate.iter().zip(&up).map(|(a, b)| gelu_tanh(*a) * b).collect();
            let y = rms(
                &lin(g("mlp.down_proj.linear.weight"), &mid, h, inter),
                Some(g("post_feedforward_layernorm.weight")),
            );
            r.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
        }
        layers.push(resid.concat());
    }

    // Pooler: padding rows are zero hidden states.
    let mut full = resid.concat();
    full.resize(total * h, 0.0);
    let (avg, cells) = pool_dense(&full, positions, d.pool_k, h);
    let root = (h as f64).sqrt();
    let pooled: Vec<f64> = avg.iter().map(|v| v * root).collect();
    // Embedder: weightless RMSNorm, then the projection.
    let proj = w("embed_vision.embedding_projection.weight");
    let soft: Vec<f64> = pooled
        .chunks_exact(h)
        .flat_map(|row| lin(proj, &rms(row, None), d.text_hidden, h))
        .collect();
    VisionRef {
        n,
        patch_embed,
        layers,
        pooled,
        soft,
        tokens: cells.len(),
    }
}
