//! Numeric parity of the GEMM lanes against a float64 reference, and the ladder
//! that runs it across every shape and operand distribution.
//!
//! ```text
//! gemm_parity score DIR [--out PATH]
//! gemm_parity ladder [--seeds N] [--shapes LABELS|all] [--dists NAMES|all]
//!                    [--binary PATH] [--external auto|none] [--python PATH] [--out PATH]
//! ```
//!
//! `score` scores one `bench_gemm_sweep --dump-parity DIR` (every dumped lane on
//! every seed, plus the comparison runtimes if `bench/gemm_sweep_mlx.py
//! --external-lanes DIR` ran on it) and prints the per-lane rows as JSON. It exits
//! non-zero on a missing lane, seed or manifest, a shape mismatch, a non-finite
//! value, a degenerate reference, a lane with no declared unit roundoff, or any
//! lane over its per-element error budget, and a budget breach names whether
//! tessl alone, the comparison runtime alone, or every runtime exceeded it,
//! because those call for opposite responses.
//!
//! `ladder` drives the dump one (shape, distribution) cell at a time (dump, run
//! the Python comparison lanes, score, delete) and merges the cells. Dumping
//! every shape at once is ~12 GB of .npy at eight draws, so peak disk is one
//! shape's worth (~3.5 GB at the widest). Why a grid and not one cell: GEMM
//! forward error grows with K, so one shape is a floor rather than a bound, and
//! the operand *distribution* moves the fraction of the error budget a lane
//! consumes by up to 15x, in the opposite direction from shape. bf16 sits at
//! 0.076x of budget on uniform operands and 0.936x on heavy-tailed ones.
//!
//! The scorer is `common/parity.rs`; `tests/gemm_parity.rs` holds its contract.
//! Exit status: 0 clean, 1 a check failed or could not run, 2 usage.

mod common;

#[path = "../json.rs"]
#[allow(dead_code)]
mod json;
#[path = "common/parity.rs"]
mod parity;
#[path = "common/pyjson.rs"]
mod pyjson;

use common::Dist;
use parity::{CellRow, LaneMerge, Precision, Scratch};
use pyjson::{fmt_e, obj, Doc};
use std::path::{Path, PathBuf};
use std::process::Command;

/// (M, N, K, label): the ladder `bench_gemm_sweep` times and dumps. A second,
/// hand-kept copy of its `SHAPES` and of `bench/gemm_sweep_mlx.py`'s; every dump
/// is cross-checked against this one (a drifted ladder would attribute every
/// number to the wrong shape), and `tests/gemm_parity.rs` compares the three
/// sources.
const LADDER: &[(usize, usize, usize, &str)] = &[
    (512, 512, 512, "square_512"),
    (1024, 1024, 1024, "square_1024"),
    (2048, 2048, 2048, "square_2048"),
    (4096, 4096, 4096, "square_4096"),
    (2048, 768, 768, "qkv_proj"),
    (8192, 3072, 768, "mlp_up"),
    (8192, 768, 3072, "mlp_down"),
    (4096, 4096, 1024, "tall_k1024"),
];

const USAGE: &str = "usage:\n  gemm_parity score DIR [--out PATH]\n  gemm_parity ladder [--seeds N] \
    [--shapes LABELS|all] [--dists NAMES|all] [--binary PATH] [--external auto|none] [--python PATH] [--out PATH]";

fn usage(msg: &str) -> ! {
    eprintln!("gemm_parity: {msg}\n{USAGE}");
    std::process::exit(2);
}

/// `--name value` or `--name=value`, consumed from `args`.
fn flag(args: &mut Vec<String>, name: &str) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == name {
            if i + 1 >= args.len() {
                usage(&format!("{name} needs a value"));
            }
            let v = args.remove(i + 1);
            args.remove(i);
            return Some(v);
        }
        if let Some(v) = args[i].strip_prefix(&format!("{name}=")) {
            let v = v.to_string();
            args.remove(i);
            return Some(v);
        }
        i += 1;
    }
    None
}

fn write_out(path: &str, text: &str) -> Result<(), String> {
    std::fs::write(path, text).map_err(|e| format!("write {path}: {e}"))
}

fn score_cmd(mut args: Vec<String>) -> Result<(), String> {
    let out = flag(&mut args, "--out");
    let [dir] = args.as_slice() else {
        usage("score takes one DIR")
    };
    let rows = parity::parity(Path::new(dir), &Precision::standard())?;
    let text = parity::rows_json(&rows)?;
    match out {
        // Written without a trailing newline, as `json.dumps` was to a file.
        Some(p) => write_out(&p, &text),
        None => {
            println!("{text}");
            Ok(())
        }
    }
}

/// How the comparison runtimes get their outputs next to a dump.
enum External {
    /// `python bench/gemm_sweep_mlx.py --external-lanes DIR`.
    Python {
        python: String,
        script: PathBuf,
    },
    None,
}

impl External {
    fn run(&self, dir: &Path) -> Result<(), String> {
        let External::Python { python, script } = self else {
            return Ok(());
        };
        let out = Command::new(python)
            .arg(script)
            .arg("--external-lanes")
            .arg(dir)
            .output()
            .map_err(|e| format!("could not run {python} {}: {e}", script.display()))?;
        eprint!("{}", String::from_utf8_lossy(&out.stderr));
        if !out.status.success() {
            return Err(format!(
                "{python} {} --external-lanes exited {}; the comparison runtimes did not run, and a \
                 partial check is not reported as a complete one (--external none runs without them)",
                script.display(),
                out.status.code().map_or("by signal".to_string(), |c| c.to_string())
            ));
        }
        Ok(())
    }
}

/// Dump one (shape, distribution), score it, delete the .npy.
fn run_cell(binary: &Path, label: &str, dist: &str, seeds: usize, external: &External) -> Result<Vec<CellRow>, String> {
    let tmp = Scratch::new(&format!("parity-{label}-{dist}"))?;
    let out = Command::new(binary)
        .arg("--dump-parity")
        .arg(&tmp.0)
        .env("BENCH_PARITY_SHAPE", label)
        .env("BENCH_PARITY_DIST", dist)
        .env("BENCH_PARITY_SEEDS", seeds.to_string())
        // Rejected outright by the dump path; drop an inherited one rather than
        // have the whole sweep die on the caller's shell state.
        .env_remove("BENCH_SHAPES")
        .output()
        .map_err(|e| format!("{label}/{dist}: could not run {}: {e}", binary.display()))?;
    if !out.status.success() {
        return Err(format!(
            "{label}/{dist}: bench_gemm_sweep exited {}\n{}",
            out.status.code().map_or("by signal".to_string(), |c| c.to_string()),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    external.run(&tmp.0)?;

    let rows = parity::parity(&tmp.0, &Precision::standard())?;

    // Cross-check the Rust ladder against this tool's copy of it. These are
    // hand-maintained lists; if they drift, every number below is being
    // attributed to the wrong shape.
    let man = parity::load_manifest(&tmp.0)?;
    let want = LADDER
        .iter()
        .find(|s| s.3 == label)
        .ok_or_else(|| format!("{label} is not in the gemm_parity ladder"))?;
    if man.shape.as_deref() != Some(label) || (man.m, man.n, man.k) != (want.0, want.1, want.2) {
        return Err(format!(
            "ladder drift: Rust dumped {} {}x{}x{}, gemm_parity expected {label} {}x{}x{}",
            man.shape.as_deref().unwrap_or("?"),
            man.m,
            man.n,
            man.k,
            want.0,
            want.1,
            want.2
        ));
    }
    // Same class of check on the distribution axis: a silently ignored
    // BENCH_PARITY_DIST would attribute uniform numbers to heavy_tail.
    if man.dist.as_deref() != Some(dist) {
        return Err(format!(
            "distribution drift: asked for '{dist}', dump reports {}",
            man.dist.as_deref().map_or("None".to_string(), |d| format!("'{d}'"))
        ));
    }
    Ok(rows
        .into_iter()
        .map(|row| CellRow {
            row,
            shape: label.to_string(),
            dist: dist.to_string(),
            m: want.0,
            n: want.1,
            k: want.2,
        })
        .collect())
}

fn names(list: Option<String>, all: Vec<String>, what: &str) -> Result<Vec<String>, String> {
    let Some(raw) = list else { return Ok(all) };
    if raw == "all" {
        return Ok(all);
    }
    let picked: Vec<String> = raw.split(',').map(|x| x.trim().to_string()).collect();
    let unknown: Vec<&String> = picked.iter().filter(|x| !all.contains(x)).collect();
    if !unknown.is_empty() {
        return Err(format!(
            "unknown {what}(s) {}; expected from {}",
            pyjson::quote_list(&unknown),
            pyjson::quote_list(&all)
        ));
    }
    Ok(picked)
}

fn ladder_cmd(mut args: Vec<String>) -> Result<(), String> {
    let seeds = match flag(&mut args, "--seeds") {
        Some(v) => v
            .parse::<usize>()
            .map_err(|_| format!("--seeds must be a positive integer, got {v:?}"))?,
        None => 8,
    };
    if seeds < 1 {
        return Err(format!("--seeds must be >= 1, got {seeds}"));
    }
    let shapes = flag(&mut args, "--shapes");
    let dists = flag(&mut args, "--dists");
    let binary = flag(&mut args, "--binary");
    let ext = flag(&mut args, "--external").unwrap_or_else(|| "auto".to_string());
    let python = flag(&mut args, "--python").unwrap_or_else(|| "python3".to_string());
    let out = flag(&mut args, "--out");
    if let Some(extra) = args.first() {
        usage(&format!("unexpected argument {extra:?}"));
    }

    let binary = match binary {
        Some(b) => PathBuf::from(b),
        None => std::env::current_exe()
            .map_err(|e| format!("current_exe: {e}"))?
            .with_file_name("bench_gemm_sweep"),
    };
    if !binary.exists() {
        return Err(format!(
            "{} not built. cargo build --release --bin bench_gemm_sweep",
            binary.display()
        ));
    }
    let external = match ext.as_str() {
        "none" => External::None,
        "auto" => {
            let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/gemm_sweep_mlx.py");
            if !script.is_file() {
                return Err(format!(
                    "{} is missing, so the comparison runtimes cannot run (--external none runs without them)",
                    script.display()
                ));
            }
            External::Python { python, script }
        }
        other => return Err(format!("--external must be auto or none, got {other:?}")),
    };

    let known: Vec<String> = LADDER.iter().map(|s| s.3.to_string()).collect();
    let labels = names(shapes, known, "ladder label")?;
    let all_dists: Vec<String> = Dist::ALL.iter().map(|d| d.name().to_string()).collect();
    let dist_names = names(dists.or_else(|| Some("uniform".to_string())), all_dists, "distribution")?;

    let cells: Vec<parity::Cell> = labels
        .iter()
        .flat_map(|l| dist_names.iter().map(move |d| (l.clone(), d.clone())))
        .collect();
    let mut per_shape: Vec<CellRow> = Vec::new();
    for (i, (label, dist)) in cells.iter().enumerate() {
        eprintln!(
            "\n===== [{}/{}] {label} / {dist} x {seeds} seeds =====",
            i + 1,
            cells.len()
        );
        per_shape.extend(run_cell(&binary, label, dist, seeds, &external)?);
    }
    let per_lane = parity::merge(&per_shape, &cells)?;

    eprintln!(
        "\n{:<20}{:<6}{:>12}{:>12}{:>14}  budget worst cell",
        "lane", "out", "worst norm", "worst elem", "worst budget"
    );
    for r in &per_lane {
        match r {
            LaneMerge::Skipped { lane, reason } => {
                let short: String = reason.chars().take(48).collect();
                eprintln!("{lane:<20}SKIPPED  {short}");
            }
            LaneMerge::Scored {
                lane,
                out_dtype,
                worst_rel_err,
                worst_elem_rel_err,
                worst_higham_ratio,
                worst_higham_shape,
                worst_higham_dist,
                ..
            } => eprintln!(
                "{lane:<20}{out_dtype:<6}{:>12}{:>12}{:>13.3}x  {worst_higham_shape}/{worst_higham_dist}",
                fmt_e(*worst_rel_err, 2),
                fmt_e(*worst_elem_rel_err, 2),
                worst_higham_ratio
            ),
        }
    }

    let doc = obj([
        ("ladder", Doc::strs(&labels)),
        ("dists", Doc::strs(&dist_names)),
        ("seeds", Doc::uint(seeds)),
        (
            "cells",
            Doc::Arr(
                cells
                    .iter()
                    .map(|(a, b)| Doc::strs(&[a.as_str(), b.as_str()]))
                    .collect(),
            ),
        ),
        ("per_lane", Doc::Arr(per_lane.iter().map(LaneMerge::doc).collect())),
        ("per_shape", Doc::Arr(per_shape.iter().map(CellRow::doc).collect())),
    ]);
    let text = doc.dumps()?;
    match out {
        Some(p) => {
            write_out(&p, &text)?;
            eprintln!("\nwrote {p}");
        }
        None => println!("{text}"),
    }
    Ok(())
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage("a subcommand is required");
    }
    let sub = args.remove(0);
    let result = match sub.as_str() {
        "score" => score_cmd(args),
        "ladder" => ladder_cmd(args),
        "-h" | "--help" => {
            println!("{USAGE}");
            Ok(())
        }
        other => usage(&format!("unknown subcommand {other:?}")),
    };
    if let Err(msg) = result {
        eprintln!("{msg}");
        std::process::exit(1);
    }
}
