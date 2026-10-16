//! What the C ABI's waits cost, through the ABI itself (`tessl::capi`), as a
//! foreign caller sees it. One process is one lane, for `bench_paired
//! --format metrics`:
//!
//! ```text
//! bench_capi [--async] [--model=SAFETENSORS[,CONFIG[,PREFIX]]] [--tokens=N]
//! paired_capi.sh ROUNDS OUT [--model=...] [--tokens=N]   # sync vs async, paired
//! ```
//!
//! Without `--async` the runtime is as `tessl_runtime_new` makes it: every
//! kernel its own waited commit. `--async` switches the runtime with
//! `tessl_runtime_set_async_encode`, as `tessl_torch` does. The default model
//! is the tiny training fixture (dispatch-bound, so it isolates the per-wait
//! cost); `--model` points at a real checkpoint (`config.json` beside it
//! unless given, prefix `model.language_model.`).
//!
//! Metrics (all medians, lower is better): `capi_idle_sync_us` is one
//! `tessl_synchronize` on an idle runtime, the wait every successful call
//! ends with; `capi_forward_us_{label}` is one `tessl_qwen35_train_forward`
//! plus its `tessl_qwen35_train_discard`; `capi_train_step_ms_{label}` is one
//! `tessl_qwen35_train_step`. `{label}` is `tiny` or `t{N}`.

mod common;

use std::ffi::{c_char, CStr, CString};
use std::path::PathBuf;
use std::ptr;
use std::time::Instant;

use tessl::capi::{
    tessl_qwen35_free, tessl_qwen35_load, tessl_qwen35_train_discard, tessl_qwen35_train_forward,
    tessl_qwen35_train_step, tessl_runtime_free, tessl_runtime_new, tessl_runtime_set_async_encode, tessl_synchronize,
    TesslQwen35, TesslRuntime, TESSL_F32, TESSL_OK, TESSL_OPERANDS_EXACT_F32, TESSL_SUPERVISE_CAUSAL,
};

const ERR_LEN: usize = 1024;
const SYNC_CALLS: usize = 400;
const FORWARD_CALLS: usize = 15;
const STEP_CALLS: usize = 5;

type Res<T> = Result<T, String>;

fn msg(err: &[c_char]) -> String {
    // SAFETY: every caller passes a buffer a tessl call NUL-terminated or left zeroed.
    unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned()
}

fn ok(what: &str, status: i32, err: &[c_char]) -> Res<()> {
    if status == TESSL_OK {
        Ok(())
    } else {
        Err(format!("{what}: status {status}: {}", msg(err)))
    }
}

fn cstr(p: &std::path::Path) -> Res<CString> {
    CString::new(p.to_str().ok_or("path is not UTF-8")?).map_err(|e| e.to_string())
}

struct Args {
    use_async: bool,
    model: Option<(PathBuf, PathBuf, String)>,
    tokens: usize,
}

fn parse() -> Res<Args> {
    let mut a = Args {
        use_async: false,
        model: None,
        tokens: 0,
    };
    for arg in std::env::args().skip(1) {
        if arg == "--async" {
            a.use_async = true;
        } else if let Some(v) = arg.strip_prefix("--model=") {
            let mut parts = v.splitn(3, ',');
            let st = PathBuf::from(parts.next().filter(|s| !s.is_empty()).ok_or("--model= needs a path")?);
            let cfg = match parts.next().filter(|s| !s.is_empty()) {
                Some(c) => PathBuf::from(c),
                None => st.parent().ok_or("--model path has no directory")?.join("config.json"),
            };
            let prefix = parts.next().unwrap_or("model.language_model.").to_string();
            a.model = Some((st, cfg, prefix));
        } else if let Some(v) = arg.strip_prefix("--tokens=") {
            a.tokens = common::parse_count("--tokens", v, 2)?;
        } else {
            return Err(format!("unknown argument {arg:?}"));
        }
    }
    Ok(a)
}

fn micros(f: impl FnMut() -> Res<()>, calls: usize) -> Res<f64> {
    let mut f = f;
    f()?;
    let mut samples = Vec::with_capacity(calls);
    for _ in 0..calls {
        let t0 = Instant::now();
        f()?;
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    common::median(samples)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("bench_capi: {e}");
        std::process::exit(1);
    }
}

fn run() -> Res<()> {
    let args = parse()?;
    let mut err = [0 as c_char; ERR_LEN];
    // SAFETY: `err` is writable for ERR_LEN bytes.
    let rt: *mut TesslRuntime = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
    if rt.is_null() {
        return Err(format!("tessl_runtime_new: {}", msg(&err)));
    }
    let on = u32::from(args.use_async);
    // SAFETY: `rt` is live on this thread.
    let s = unsafe { tessl_runtime_set_async_encode(rt, on, err.as_mut_ptr(), ERR_LEN) };
    ok("tessl_runtime_set_async_encode", s, &err)?;

    let sync_us = micros(
        || {
            let mut err = [0 as c_char; ERR_LEN];
            // SAFETY: `rt` is live on this thread.
            let s = unsafe { tessl_synchronize(rt, err.as_mut_ptr(), ERR_LEN) };
            ok("tessl_synchronize", s, &err)
        },
        SYNC_CALLS,
    )?;
    println!("METRIC capi_idle_sync_us {sync_us:.3}");

    let (st, cfg, prefix, label, ids) = match &args.model {
        None => {
            let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train");
            let ids: Vec<u32> = tessl::npy::read_npy(&dir.join("ids.npy"))?
                .i64_slice()?
                .iter()
                .map(|&x| x as u32)
                .collect();
            (
                dir.join("model.safetensors"),
                dir.join("config.json"),
                "model.".to_string(),
                "tiny".to_string(),
                ids,
            )
        }
        Some((st, cfg, prefix)) => {
            if args.tokens == 0 {
                return Err("--model needs --tokens=N".into());
            }
            let ids = (0..args.tokens as u32).map(|i| (i * 104_729 + 17) % 100_000).collect();
            (
                st.clone(),
                cfg.clone(),
                prefix.clone(),
                format!("t{}", args.tokens),
                ids,
            )
        }
    };
    let (st_c, cfg_c, prefix_c) = (
        cstr(&st)?,
        cstr(&cfg)?,
        CString::new(prefix).map_err(|e| e.to_string())?,
    );
    let mut model: *mut TesslQwen35 = ptr::null_mut();
    // SAFETY: NUL-terminated strings, writable `model` and `err`.
    let s = unsafe {
        tessl_qwen35_load(
            rt,
            st_c.as_ptr(),
            cfg_c.as_ptr(),
            prefix_c.as_ptr(),
            TESSL_F32,
            &mut model,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    ok("tessl_qwen35_load", s, &err)?;

    let forward_us = micros(
        || {
            let mut err = [0 as c_char; ERR_LEN];
            let mut loss = 0.0f64;
            // SAFETY: `model` is live on this thread; `ids` has `ids.len()` readable u32s.
            let s = unsafe {
                tessl_qwen35_train_forward(
                    model,
                    ids.as_ptr(),
                    ids.len() as u64,
                    TESSL_OPERANDS_EXACT_F32,
                    TESSL_SUPERVISE_CAUSAL,
                    ptr::null(),
                    ptr::null(),
                    0,
                    1.0,
                    &mut loss,
                    err.as_mut_ptr(),
                    ERR_LEN,
                )
            };
            ok("tessl_qwen35_train_forward", s, &err)?;
            // SAFETY: as above.
            let s = unsafe { tessl_qwen35_train_discard(model, err.as_mut_ptr(), ERR_LEN) };
            ok("tessl_qwen35_train_discard", s, &err)
        },
        FORWARD_CALLS,
    )?;
    println!("METRIC capi_forward_us_{label} {forward_us:.3}");

    let step_calls = if args.model.is_some() { 3 } else { STEP_CALLS * 4 };
    let step_ms = micros(
        || {
            let mut err = [0 as c_char; ERR_LEN];
            let mut loss = 0.0f64;
            // SAFETY: as above.
            let s = unsafe {
                tessl_qwen35_train_step(
                    model,
                    ids.as_ptr(),
                    ids.len() as u64,
                    TESSL_OPERANDS_EXACT_F32,
                    &mut loss,
                    err.as_mut_ptr(),
                    ERR_LEN,
                )
            };
            ok("tessl_qwen35_train_step", s, &err)?;
            if !loss.is_finite() {
                return Err(format!("loss {loss} is not finite"));
            }
            Ok(())
        },
        step_calls,
    )? / 1e3;
    println!("METRIC capi_train_step_ms_{label} {step_ms:.4}");

    // SAFETY: both handles are live and freed once, on this thread.
    unsafe {
        if tessl_qwen35_free(model) != TESSL_OK || tessl_runtime_free(rt) != TESSL_OK {
            return Err("freeing the handles failed".into());
        }
    }
    Ok(())
}
