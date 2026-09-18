#!/usr/bin/env python3
"""Apply Wave 1+2 nn.rs hardening to tessl 0.1.4. Idempotent checks; fails loud."""
from __future__ import annotations

import re
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "src" / "nn.rs"


def main() -> None:
    text = PATH.read_text()
    assert text.count("{") == text.count("}")
    assert "elems_product" not in text, "already transformed?"

    # --- docs ---
    old_doc = """//! # The `_with_scalars` seam
//!
//! Each kernel has two entry points. The plain one binds its scalar operands
//! through the runtime's const arena and is what most callers want. The
//! `_with_scalars` one takes a closure that binds them itself, for callers
//! that need *stable* GPU addresses across encodes — const-arena offsets move
//! from one encode to the next, which breaks an Indirect Command Buffer that
//! froze its binds. `gemma-metal` drives these from a persistent scalar pool
//! for exactly that reason.
//!
//! The closure receives the binder and must fill the scalar indices named in
//! each function's docs. Buffer operands and the dispatch shape are bound here,
//! so the two paths cannot disagree about them."""
    new_doc = """//! # Safety of the `_with_scalars` seam
//!
//! Each kernel has two entry points. The plain one binds its scalar operands
//! through the runtime's const arena and is what most callers want. The
//! `unsafe` `_with_scalars` one takes a closure that binds them itself, for callers
//! that need *stable* GPU addresses across encodes — const-arena offsets move
//! from one encode to the next, which breaks an Indirect Command Buffer that
//! froze its binds. `gemma-metal` drives these from a persistent scalar pool
//! for exactly that reason.
//!
//! The closure receives the full [`Binder`], so the type system cannot stop it
//! from replacing a validated data-buffer bind, changing the pipeline, or
//! binding a device scalar whose contents disagree with the host values used
//! for validation and dispatch. A violation can turn a safe-looking call into
//! an out-of-bounds GPU access. Every `_with_scalars` caller must therefore:
//!
//! - bind every scalar index documented by that function exactly once, with the
//!   documented ABI type and the exact value passed to the host wrapper;
//! - bind no other index and perform no pipeline, dispatch, ICB, or resource
//!   mutation through the supplied binder; and
//! - ensure every buffer used as stable scalar storage belongs to `rt`, covers
//!   the bound value at its offset, and remains alive and resident until the
//!   encoded work completes (or through every replay of an ICB that froze it).
//!
//! Prefer the plain safe entry points unless stable addresses are required.
//!
//! ```compile_fail
//! use std::sync::Arc;
//! use tessl::{nn, GpuBuffer, GpuRuntime};
//!
//! fn invalid_safe_call(rt: &Arc<GpuRuntime>, x: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer) {
//!     nn::rms_norm_f32_with_scalars(rt, x, w, out, 1, 1, |_| {});
//! }
//! ```"""
    assert old_doc in text
    text = text.replace(old_doc, new_doc)

    old_helpers = """/// `rows * dim`, or an error naming the overflow rather than wrapping.
fn elems(rows: u32, dim: u32, what: &str) -> Result<usize, String> {
    (rows as usize)
        .checked_mul(dim as usize)
        .ok_or_else(|| format!("{what}: rows {rows} x dim {dim} overflows usize"))
}

/// Reject a buffer that cannot hold `need` elements of `T`.
fn require<T>(buf: &GpuBuffer, need: usize, what: &str) -> Result<(), String> {
    let have = capacity_of::<T>(buf);
    if have < need {
        return Err(format!(
            "{what}: buffer holds {have} elements, kernel reads/writes {need}"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------- RMSNorm ---"""

    new_helpers = Path(__file__).with_name("wave2_nn_helpers.rs.txt")
    # inline helpers below
    new_helpers = r'''/// `rows * dim`, or an error naming the overflow rather than wrapping.
fn elems(rows: u32, dim: u32, what: &str) -> Result<usize, String> {
    elems_product(&[rows, dim], what)
}

/// Product of device `u32` dimensions, widened before every multiplication.
fn elems_product(dims: &[u32], what: &str) -> Result<usize, String> {
    dims.iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim as usize))
        .ok_or_else(|| format!("{what}: dimension product overflows usize"))
}

fn require_runtime(rt: &GpuRuntime, buf: &GpuBuffer, what: &str) -> Result<(), String> {
    if !buf.belongs_to(rt) {
        return Err(format!("{what}: buffer belongs to another runtime"));
    }
    Ok(())
}

fn require_capacity<T>(buf: &GpuBuffer, need: usize, what: &str) -> Result<(), String> {
    let have = capacity_of::<T>(buf);
    if have < need {
        return Err(format!(
            "{what}: buffer holds {have} elements, kernel reads/writes {need}"
        ));
    }
    Ok(())
}

fn require<T>(rt: &GpuRuntime, buf: &GpuBuffer, need: usize, what: &str) -> Result<(), String> {
    require_runtime(rt, buf, what)?;
    require_capacity::<T>(buf, need, what)
}

fn require_disjoint_writes(
    entry: &str,
    writes: &[(&str, &GpuBuffer)],
    reads: &[(&str, &GpuBuffer)],
) -> Result<(), String> {
    for (i, &(lhs_name, lhs)) in writes.iter().enumerate() {
        for &(rhs_name, rhs) in &writes[i + 1..] {
            if lhs.aliases(rhs) {
                return Err(format!(
                    "{entry}: writable buffers {lhs_name} and {rhs_name} overlap"
                ));
            }
        }
        for &(read_name, read) in reads {
            if lhs.aliases(read) {
                return Err(format!(
                    "{entry}: writable buffer {lhs_name} overlaps read-only buffer {read_name}"
                ));
            }
        }
    }
    Ok(())
}

fn validate_rms_scalars(dim: u32, eps: f32, what: &str) -> Result<(), String> {
    if dim == 0 {
        return Err(format!("{what}: dim must be non-zero"));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(format!("{what}: eps must be finite and positive"));
    }
    Ok(())
}

pub fn attn_kv_capacity(
    k: &GpuBuffer,
    v: &GpuBuffer,
    batch: u32,
    heads_kv: u32,
    head_dim: u32,
) -> Result<u32, String> {
    attn_kv_capacity_for(k, v, batch, heads_kv, head_dim, "attention")
}

fn attn_kv_capacity_for(
    k: &GpuBuffer,
    v: &GpuBuffer,
    batch: u32,
    heads_kv: u32,
    head_dim: u32,
    what: &str,
) -> Result<u32, String> {
    if heads_kv == 0 {
        return Err(format!("{what}: heads_kv must be non-zero"));
    }
    if head_dim == 0 {
        return Err(format!("{what}: head_dim must be non-zero"));
    }
    let per_position = elems_product(&[batch, heads_kv, head_dim], what)?;
    if per_position == 0 {
        return Ok(0);
    }
    let k_positions = capacity_of::<f32>(k) / per_position;
    let v_positions = capacity_of::<f32>(v) / per_position;
    if k_positions != v_positions {
        return Err(format!(
            "{what}: K and V imply different fixed capacities ({k_positions} vs \
             {v_positions} positions)"
        ));
    }
    u32::try_from(k_positions)
        .map_err(|_| format!("{what}: KV capacity {k_positions} exceeds the device u32 range"))
}

pub fn validate_attn_storage(
    dims: &AttnDims,
    d: u32,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    out_bf16: bool,
) -> Result<u32, String> {
    validate_attn_storage_for(dims, d, q, k, v, o, "attention", out_bf16)
}

fn validate_attn_storage_for(
    dims: &AttnDims,
    d: u32,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    what: &str,
    out_bf16: bool,
) -> Result<u32, String> {
    validate_attn_dims(dims, d, q, o, what, out_bf16)?;
    let capacity = attn_kv_capacity_for(k, v, dims.batch, dims.heads_kv, d, what)?;
    let has_work = dims.batch != 0 && dims.tq != 0 && dims.heads != 0;
    if !has_work {
        return Ok(capacity);
    }
    if capacity == 0 {
        return Err(format!(
            "{what}: K/V buffers do not jointly back one complete KV position"
        ));
    }
    Ok(capacity)
}

// ---------------------------------------------------------------- RMSNorm ---'''
    assert old_helpers in text
    text = text.replace(old_helpers, new_helpers)

    # require rewrite only inside fns that take rt
    def rewrite_requires(src: str) -> str:
        out = []
        i = 0
        for m in re.finditer(r"(?m)^(pub (?:unsafe )?fn \w+\([\s\S]*?\)[^{]*\{)", src):
            out.append(src[i : m.start()])
            header = m.group(1)
            # find matching close brace for function — approximate: until next pub fn at column 0
            start = m.end()
            nxt = re.search(r"(?m)^pub (?:unsafe )?fn ", src[start:])
            end = start + nxt.start() if nxt else len(src)
            body = src[start:end]
            if "rt:" in header or "rt: &" in header:
                body = re.sub(r"require::<([^>]+)>\((?!rt,)", r"require::<\1>(rt, ", body)
            out.append(header + body)
            i = end
        out.append(src[i:])
        return "".join(out)

    text = rewrite_requires(text)

    # bank validates need rt param
    text = text.replace(
        "fn validate(&self, shape: &QuantShape, what: &str) -> Result<(), String> {",
        "fn validate(&self, rt: &GpuRuntime, shape: &QuantShape, what: &str) -> Result<(), String> {",
    )
    # call sites bank.validate(& -> bank.validate(rt, &
    def fix_bank_calls(src: str) -> str:
        out = []
        i = 0
        for m in re.finditer(r"(\w+)\.validate\(&", src):
            name = m.group(1)
            if name in ("shape", "dims", "self"):
                out.append(src[i : m.end()])
                i = m.end()
                continue
            out.append(src[i : m.start()])
            out.append(f"{name}.validate(rt, &")
            i = m.end()
        out.append(src[i:])
        return "".join(out)

    text = fix_bank_calls(text)

    # validate_attn_dims uses require_capacity (no rt)
    text = text.replace(
        "let n = elems(dims.batch * dims.tq * dims.heads, d, what)?;",
        "let n = elems_product(&[dims.batch, dims.tq, dims.heads, d], what)?;",
    )
    text = text.replace(
        'require::<f32>(rt, q, n, &format!("{what} q"))?;',
        'require_capacity::<f32>(q, n, &format!("{what} q"))?;',
    )
    text = text.replace(
        'require::<u16>(rt, o, n, &format!("{what} o (bf16)"))?;',
        'require_capacity::<u16>(o, n, &format!("{what} o (bf16)"))?;',
    )
    text = text.replace(
        'require::<f32>(rt, o, n, &format!("{what} o"))?;',
        'require_capacity::<f32>(o, n, &format!("{what} o"))?;',
    )

    # flash attn KV storage
    text = text.replace(
        """    validate_attn_dims(&dims, d, q, o, "flash_attn_swa", false)?;
    require::<u32>(rt, tkv, 1, "flash_attn_swa tkv")?;
    require::<u32>(rt, q_pos_offset, 1, "flash_attn_swa q_pos_offset")?;
    require::<u32>(rt, kv_pos_offset, 1, "flash_attn_swa kv_pos_offset")?;
    // K/V hold at least one position each; `Tkv` lives on the device.
    require::<f32>(
        rt, k,
        elems(dims.batch * dims.heads_kv, d, "flash_attn_swa k")?,
        "flash_attn_swa k",
    )?;
    require::<f32>(
        rt, v,
        elems(dims.batch * dims.heads_kv, d, "flash_attn_swa v")?,
        "flash_attn_swa v",
    )?;""",
        """    let _kv_capacity =
        validate_attn_storage_for(&dims, d, q, k, v, o, "flash_attn_swa", false)?;
    require::<u32>(rt, tkv, 1, "flash_attn_swa tkv")?;
    require::<u32>(rt, q_pos_offset, 1, "flash_attn_swa q_pos_offset")?;
    require::<u32>(rt, kv_pos_offset, 1, "flash_attn_swa kv_pos_offset")?;
    require_disjoint_writes(
        "flash_attn_swa",
        &[("o", o)],
        &[("q", q), ("k", k), ("v", v), ("tkv", tkv)],
    )?;""",
    )
    # tolerate alternate require formatting from rewrite
    if 'validate_attn_dims(&dims, d, q, o, "flash_attn_swa"' in text:
        # try with rt, \n formatting
        text = re.sub(
            r'    validate_attn_dims\(&dims, d, q, o, "flash_attn_swa", false\)\?;\n'
            r'    require::<u32>\(rt, tkv, 1, "flash_attn_swa tkv"\)\?;\n'
            r'    require::<u32>\(rt, q_pos_offset, 1, "flash_attn_swa q_pos_offset"\)\?;\n'
            r'    require::<u32>\(rt, kv_pos_offset, 1, "flash_attn_swa kv_pos_offset"\)\?;\n'
            r'    // K/V hold at least one position each; `Tkv` lives on the device\.\n'
            r'    require::<f32>\(rt,\s*\n?\s*k,\n'
            r'        elems\(dims\.batch \* dims\.heads_kv, d, "flash_attn_swa k"\)\?,\n'
            r'        "flash_attn_swa k",\n'
            r'    \)\?;\n'
            r'    require::<f32>\(rt,\s*\n?\s*v,\n'
            r'        elems\(dims\.batch \* dims\.heads_kv, d, "flash_attn_swa v"\)\?,\n'
            r'        "flash_attn_swa v",\n'
            r'    \)\?;',
            """    let _kv_capacity =
        validate_attn_storage_for(&dims, d, q, k, v, o, "flash_attn_swa", false)?;
    require::<u32>(rt, tkv, 1, "flash_attn_swa tkv")?;
    require::<u32>(rt, q_pos_offset, 1, "flash_attn_swa q_pos_offset")?;
    require::<u32>(rt, kv_pos_offset, 1, "flash_attn_swa kv_pos_offset")?;
    require_disjoint_writes(
        "flash_attn_swa",
        &[("o", o)],
        &[("q", q), ("k", k), ("v", v), ("tkv", tkv)],
    )?;""",
            text,
            count=1,
        )

    text = re.sub(
        r'    validate_attn_dims\(&dims, D, q, o, "flash_attn_global_h512", out_bf16\)\?;\n'
        r'    require::<u32>\(rt, tkv, 1, "flash_attn_global_h512 tkv"\)\?;\n'
        r'    require::<u32>\(rt, q_pos_offset, 1, "flash_attn_global_h512 q_pos_offset"\)\?;\n'
        r'    require::<u32>\(rt, kv_pos_offset, 1, "flash_attn_global_h512 kv_pos_offset"\)\?;\n'
        r'    require::<f32>\(rt,\s*\n?\s*k,\n'
        r'        elems\(dims\.batch \* dims\.heads_kv, D, "k"\)\?,\n'
        r'        "flash_attn_global_h512 k",\n'
        r'    \)\?;\n'
        r'    require::<f32>\(rt,\s*\n?\s*v,\n'
        r'        elems\(dims\.batch \* dims\.heads_kv, D, "v"\)\?,\n'
        r'        "flash_attn_global_h512 v",\n'
        r'    \)\?;',
        """    let _kv_capacity = validate_attn_storage_for(
        &dims, D, q, k, v, o, "flash_attn_global_h512", out_bf16,
    )?;
    require::<u32>(rt, tkv, 1, "flash_attn_global_h512 tkv")?;
    require::<u32>(rt, q_pos_offset, 1, "flash_attn_global_h512 q_pos_offset")?;
    require::<u32>(rt, kv_pos_offset, 1, "flash_attn_global_h512 kv_pos_offset")?;
    require_disjoint_writes(
        "flash_attn_global_h512",
        &[("o", o)],
        &[("q", q), ("k", k), ("v", v), ("tkv", tkv)],
    )?;""",
        text,
        count=1,
    )

    # unsafe with_scalars
    text, n = re.subn(
        r"^pub fn (\w+_with_scalars)\(",
        r"pub unsafe fn \1(",
        text,
        flags=re.M,
    )
    print(f"unsafe fns: {n}")

    # wrap calls (skip defs and doc comments)
    def line_at(src: str, pos: int) -> str:
        ls = src.rfind("\n", 0, pos) + 1
        le = src.find("\n", pos)
        return src[ls : le if le >= 0 else len(src)]

    sites = []
    for m in re.finditer(r"\b(\w+_with_scalars)\(", text):
        before = text[max(0, m.start() - 24) : m.start()]
        if re.search(r"(?:unsafe\s+)?fn\s+$", before):
            continue
        line = line_at(text, m.start())
        if line.lstrip().startswith(("//!", "///", "*")):
            continue
        window = text[max(0, m.start() - 80) : m.start()]
        if re.search(r"unsafe\s*\{[^}]*$", window):
            continue
        j = m.end()
        depth = 1
        while j < len(text) and depth:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        sites.append((m.start(), j))
    for start, end in reversed(sites):
        call = text[start:end]
        ls = text.rfind("\n", 0, start) + 1
        indent = re.match(r"[ \t]*", text[ls:]).group(0)
        if "\n" in call:
            wrapped = (
                "unsafe {\n"
                + indent
                + "    "
                + call.replace("\n", "\n    ")
                + "\n"
                + indent
                + "}"
            )
        else:
            wrapped = f"unsafe {{ {call} }}"
        text = text[:start] + wrapped + text[end:]
    print(f"wrapped calls: {len(sites)}")

    # rms validate inject
    for fname in ("rms_norm_f32", "rms_norm_bf16", "rms_norm_residual_add_f32"):
        m = re.search(
            rf"pub fn {fname}\([\s\S]*?\) -> Result<\(\), String> \{{\n",
            text,
        )
        assert m, fname
        after = text[m.end() : m.end() + 40]
        assert after.lstrip().startswith("unsafe"), (fname, after)
        text = (
            text[: m.end()]
            + f'    validate_rms_scalars(dim, eps, "{fname}")?;\n'
            + "    // SAFETY: binds only documented scalar slots with validated host values.\n"
            + text[m.end() :]
        )

    # Alias inserts: after requires, after zero-work early return when present
    alias_specs = [
        (
            'require::<f32>(rt, out, n, "rms_norm_f32 out")?;',
            'require_disjoint_writes("rms_norm_f32", &[("out", out)], &[("weight", weight)])?;',
            True,
        ),
        (
            'require::<u16>(rt, out, n, "rms_norm_bf16 out")?;',
            '''require_disjoint_writes(
        "rms_norm_bf16",
        &[("out", out)],
        &[("x", x), ("weight", weight)],
    )?;''',
            True,
        ),
        (
            'require::<f32>(rt, resid, n, "rms_norm_residual_add_f32 resid")?;',
            '''require_disjoint_writes(
        "rms_norm_residual_add_f32",
        &[("resid", resid)],
        &[("weight", weight)],
    )?;''',
            True,
        ),
        (
            'require::<u16>(rt, out, n_us, "mlp_gelu_tanh_bf16 out")?;',
            '''require_disjoint_writes(
        "mlp_gelu_tanh_bf16",
        &[("out", out)],
        &[("gate", gate), ("up", up)],
    )?;''',
            False,
        ),
        (
            'require::<f32>(rt, y, rows as usize, "gemv_q8 y")?;',
            '''require_disjoint_writes(
        "gemv_q8",
        &[("y", y)],
        &[("packed", packed), ("scales", scales), ("zeros", zeros), ("x", x)],
    )?;''',
            False,
        ),
        (
            'require::<u32>(rt, dst_offset, 1, "kv_store_timestep dst_offset")?;',
            '''require_disjoint_writes(
        "kv_store_timestep",
        &[("dst", dst)],
        &[("src", src), ("dst_offset", dst_offset)],
    )?;''',
            False,
        ),
        (
            'require::<u32>(rt, dst_offset, 1, "kv_store_timestep_pair dst_offset")?;',
            '''require_disjoint_writes(
        "kv_store_timestep_pair",
        &[("dst_k", dst_k), ("dst_v", dst_v)],
        &[("src_k", src_k), ("src_v", src_v), ("dst_offset", dst_offset)],
    )?;''',
            False,
        ),
        (
            'require::<u32>(rt, start, 1, "kv_ring_densify start")?;',
            '''require_disjoint_writes(
        "kv_ring_densify",
        &[("dst", dst)],
        &[("src", src), ("filled", filled), ("start", start)],
    )?;''',
            False,
        ),
        (
            'require::<f32>(rt, softcap, 1, "softcap_logits softcap")?;',
            '''require_disjoint_writes(
        "softcap_logits",
        &[("logits", logits)],
        &[("softcap", softcap)],
    )?;''',
            False,
        ),
        (
            'require::<f32>(rt, softcap, 1, "softcap_sample softcap")?;',
            '''require_disjoint_writes(
        "softcap_sample",
        &[("logits", logits), ("out_token", out_token)],
        &[("softcap", softcap)],
    )?;''',
            False,
        ),
        (
            'require::<f32>(rt, softcap, 1, "softcap_argmax_one_pass softcap")?;',
            '''require_disjoint_writes(
        "softcap_argmax_one_pass",
        &[("out_token", out_token)],
        &[("logits", logits), ("softcap", softcap)],
    )?;''',
            False,
        ),
    ]

    for needle, block, after_zero_rows in alias_specs:
        idx = text.find(needle)
        if idx < 0:
            # try multiline require form
            print("MISS needle", needle[:50])
            continue
        insert_at = idx + len(needle)
        # skip past following early return if after_zero_rows
        rest = text[insert_at : insert_at + 120]
        if after_zero_rows and "if rows == 0" in rest:
            # find end of early return block
            m = re.match(
                r"\n    if rows == 0 \{\n        return Ok\(\(\);\n    \}",
                text[insert_at:],
            )
            if m:
                insert_at += m.end()
        if "require_disjoint_writes" in text[insert_at : insert_at + 80]:
            continue
        text = text[:insert_at] + "\n    " + block + text[insert_at:]
        print("alias", needle[20:40])

    # argmax special
    if "argmax_f32_pass_with_scalars" in text and 'match idx_in {\n        Some(indices) => require_disjoint' not in text:
        needle = 'if let Some(b) = idx_in {\n        require::<u32>(rt, b, n as usize, "argmax_f32_pass idx_in")?;\n    }'
        block = '''
    match idx_in {
        Some(indices) => require_disjoint_writes(
            "argmax_f32_pass",
            &[("out_idx", out_idx), ("out_val", out_val)],
            &[("logits", logits), ("idx_in", indices), ("softcap", softcap)],
        )?,
        None => require_disjoint_writes(
            "argmax_f32_pass",
            &[("out_idx", out_idx), ("out_val", out_val)],
            &[("logits", logits), ("softcap", softcap)],
        )?,
    }'''
        if needle in text:
            text = text.replace(needle, needle + block, 1)
            print("alias argmax")

    # More q4 / embed / gate / kv / gemm aliases — insert before pipeline when missing
    more = {
        "gemv_q4_with_scalars": '''    require_disjoint_writes(
        "gemv_q4",
        &[("y", y)],
        &[("packed", bank.packed), ("scales", bank.scales), ("zeros", bank.zeros), ("x", x)],
    )?;
''',
        "embed_lookup_q4_with_scalars": '''    require_disjoint_writes(
        "embed_lookup_q4",
        &[("out", out)],
        &[("packed", bank.packed), ("scales", bank.scales), ("zeros", bank.zeros), ("token_ids", token_ids)],
    )?;
''',
        "embed_lookup_q4_mlx_with_scalars": '''    require_disjoint_writes(
        "embed_lookup_q4_mlx",
        &[("out", out)],
        &[("packed", bank.packed), ("scales_biases", bank.scales_biases), ("token_ids", token_ids)],
    )?;
''',
        "gemv_q4_mlx_with_scalars": '''    require_disjoint_writes(
        "gemv_q4_mlx",
        &[("y", y)],
        &[("packed", bank.packed), ("scales_biases", bank.scales_biases), ("x", x)],
    )?;
''',
        "gemv_q4_mlx_blocked_with_scalars": '''    require_disjoint_writes(
        "gemv_q4_mlx_blocked",
        &[("y", y)],
        &[("packed", bank.packed), ("scales_biases", bank.scales_biases), ("x", x)],
    )?;
''',
        "gemv_q4_mlx_simd_with_scalars": '''    require_disjoint_writes(
        "gemv_q4_mlx_simd",
        &[("y", y)],
        &[("packed", bank.packed), ("scales_biases", bank.scales_biases), ("x_bf16", x_bf16)],
    )?;
''',
        "gemv_q4_mlx_gate_up_gelu_with_scalars": '''    require_disjoint_writes(
        "gemv_q4_mlx_gate_up_gelu",
        &[("mid", mid)],
        &[
            ("gate_packed", gate.packed),
            ("gate_scales_biases", gate.scales_biases),
            ("up_packed", up.packed),
            ("up_scales_biases", up.scales_biases),
            ("x", x),
        ],
    )?;
''',
        "gemv_q4_mlx_kv_with_scalars": '''    require_disjoint_writes(
        "gemv_q4_mlx_kv",
        &[("k_out", k_out), ("v_out", v_out)],
        &[
            ("k_packed", k.packed),
            ("k_scales_biases", k.scales_biases),
            ("v_packed", v.packed),
            ("v_scales_biases", v.scales_biases),
            ("x_bf16", x_bf16),
        ],
    )?;
''',
        "gemm_q4_mlx_with_scalars": '''    require_disjoint_writes(
        "gemm_q4_mlx",
        &[("y", y)],
        &[("packed", bank.packed), ("scales_biases", bank.scales_biases), ("x_bf16", x_bf16)],
    )?;
''',
    }
    for fname, block in more.items():
        m = re.search(rf"pub unsafe fn {fname}\(", text)
        if not m:
            print("missing fn", fname)
            continue
        start = m.start()
        nxt = re.search(r"\npub (?:unsafe )?fn ", text[start + 20 :])
        end = start + 20 + nxt.start() if nxt else start + 4000
        body = text[start:end]
        if "require_disjoint_writes" in body:
            continue
        am = re.search(r"    let p = rt\.pipeline", body)
        if not am:
            print("no pipeline", fname)
            continue
        at = start + am.start()
        text = text[:at] + block + text[at:]
        print("more", fname)

    # gemm_i8
    if "pub fn gemm_i8_dequant" in text and 'require_disjoint_writes(\n            "gemm_i8_dequant"' not in text:
        m = re.search(r"pub fn gemm_i8_dequant\(", text)
        start = m.start()
        nxt = re.search(r"\npub (?:unsafe )?fn |\nfn ", text[start + 20 :])
        end = start + 20 + nxt.start() if nxt else start + 3000
        body = text[start:end]
        if "require_disjoint_writes" not in body:
            am = re.search(r"    let p = rt\.pipeline", body)
            if am:
                block = '''    match b_scale {
        Some(sc) => require_disjoint_writes(
            "gemm_i8_dequant",
            &[("c", c)],
            &[("a", a), ("b", b), ("b_scale", sc)],
        )?,
        None => require_disjoint_writes(
            "gemm_i8_dequant",
            &[("c", c)],
            &[("a", a), ("b", b)],
        )?,
    }
'''
                at = start + am.start()
                text = text[:at] + block + text[at:]
                print("more gemm_i8")

    # rms_qkv_rope aliases
    if "pub unsafe fn rms_qkv_rope_with_scalars" in text:
        m = re.search(r"pub unsafe fn rms_qkv_rope_with_scalars\(", text)
        start = m.start()
        nxt = re.search(r"\npub (?:unsafe )?fn ", text[start + 20 :])
        end = start + 20 + nxt.start() if nxt else start + 5000
        body = text[start:end]
        if "require_disjoint_writes" not in body:
            am = re.search(r"    let p = rt\.pipeline", body)
            assert am
            block = '''    if q_only {
        match pos_offset_buf {
            Some(pos) => require_disjoint_writes(
                "rms_qkv_rope",
                &[("q", qkv.q)],
                &[("q_weight", qkv.q_weight), ("pos_offset_buf", pos)],
            )?,
            None => require_disjoint_writes(
                "rms_qkv_rope",
                &[("q", qkv.q)],
                &[("q_weight", qkv.q_weight)],
            )?,
        }
    } else {
        match (pos_offset_buf, kv_store) {
            (None, None) => require_disjoint_writes(
                "rms_qkv_rope",
                &[("q", qkv.q), ("k", qkv.k), ("v", qkv.v)],
                &[("q_weight", qkv.q_weight), ("k_weight", qkv.k_weight), ("v_weight", qkv.v_weight)],
            )?,
            (Some(pos), None) => require_disjoint_writes(
                "rms_qkv_rope",
                &[("q", qkv.q), ("k", qkv.k), ("v", qkv.v)],
                &[
                    ("q_weight", qkv.q_weight),
                    ("k_weight", qkv.k_weight),
                    ("v_weight", qkv.v_weight),
                    ("pos_offset_buf", pos),
                ],
            )?,
            (Some(pos), Some(target)) => require_disjoint_writes(
                "rms_qkv_rope",
                &[("q", qkv.q), ("k", qkv.k), ("v", qkv.v), ("dst_k", target.dst_k), ("dst_v", target.dst_v)],
                &[
                    ("q_weight", qkv.q_weight),
                    ("k_weight", qkv.k_weight),
                    ("v_weight", qkv.v_weight),
                    ("pos_offset_buf", pos),
                    ("dst_offset", target.dst_offset),
                ],
            )?,
            (None, Some(target)) => require_disjoint_writes(
                "rms_qkv_rope",
                &[("q", qkv.q), ("k", qkv.k), ("v", qkv.v), ("dst_k", target.dst_k), ("dst_v", target.dst_v)],
                &[
                    ("q_weight", qkv.q_weight),
                    ("k_weight", qkv.k_weight),
                    ("v_weight", qkv.v_weight),
                    ("dst_offset", target.dst_offset),
                ],
            )?,
        }
    }
'''
            at = start + am.start()
            text = text[:at] + block + text[at:]
            print("more rms_qkv")

    assert text.count("{") == text.count("}"), text.count("{") - text.count("}")
    assert "pub unsafe fn unsafe" not in text
    assert "nn::unsafe" not in text
    assert "elems_product" in text
    PATH.write_text(text)
    print("SUCCESS", PATH, PATH.stat().st_size)


if __name__ == "__main__":
    main()
