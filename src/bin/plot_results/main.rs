//! Draw the benchmark figures under `bench/results/` as SVG from the result
//! files behind them. Analysis only; ships nothing.
//!
//! ```text
//! plot_results NAME RESULT.json [--out-dir DIR]   draw one figure
//! plot_results all RESULTS_DIR  [--out-dir DIR]   draw every figure from its file in RESULTS_DIR
//! ```
//!
//! The figure is written as `<stem of RESULT.json>.svg` next to the result file
//! (or in `DIR`), and that is printed. Figures that need sibling files (the
//! tokenizer figure reads the two lanes' own output, the vision figure reads
//! the vision profile) find them in the same directory. Each figure replaces
//! one matplotlib script and keeps its panels, series, axes and refusals; the
//! chart writer is `chart.rs`, no dependency, and the same data is the same
//! bytes. Where a script would have raised (a missing key, an empty list, lanes
//! that disagree on the token ids) this exits 1 with the reason and writes
//! nothing.

// The crate's strict parser, shared the way bench_paired shares it; this tool
// does not use its object lookup.
#[allow(dead_code)]
#[path = "../../json.rs"]
mod json;

mod chart;
mod data;
mod media;
mod memory;
mod paired;
mod registry;
mod small_m;
mod stage;
mod svg;
mod tok;
mod vision;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str =
    "usage: plot_results NAME RESULT.json [--out-dir DIR]\n       plot_results all RESULTS_DIR [--out-dir DIR]";

struct Args {
    name: String,
    input: PathBuf,
    out_dir: Option<PathBuf>,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut positional = Vec::new();
    let mut out_dir = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--out-dir" {
            let dir = it.next().ok_or("--out-dir needs a directory")?;
            if out_dir.replace(PathBuf::from(dir)).is_some() {
                return Err("--out-dir given twice".to_string());
            }
        } else if a.starts_with("--") {
            return Err(format!("unknown flag {a:?}\n{USAGE}"));
        } else {
            positional.push(a.clone());
        }
    }
    match positional.as_slice() {
        [name, input] => Ok(Args {
            name: name.clone(),
            input: PathBuf::from(input),
            out_dir,
        }),
        _ => Err(format!(
            "expected a figure name and a path\n{USAGE}\nfigures: {}",
            registry::names()
        )),
    }
}

/// Write a drawn figure as `<stem of input>.svg` into `out_dir` (default: beside `input`).
fn write(out: registry::Output, input: &Path, out_dir: Option<&Path>) -> Result<(), String> {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("{}: not a file name", input.display()))?;
    let dir = match out_dir {
        Some(d) => d.to_path_buf(),
        None => input.parent().map(Path::to_path_buf).unwrap_or_default(),
    };
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(format!("{stem}.svg"));
    std::fs::write(&path, out.svg).map_err(|e| format!("{}: {e}", path.display()))?;
    print!("{}", out.table);
    println!("wrote {}", path.display());
    Ok(())
}

fn run(args: &[String]) -> Result<(), String> {
    let args = parse_args(args)?;
    if args.name == "all" {
        // Every figure is drawn before any is written, so a refusal leaves the directory as it was.
        let mut drawn = Vec::new();
        for plot in registry::PLOTS {
            let input = args.input.join(plot.file);
            let out = (plot.render)(&input).map_err(|e| format!("{}: {e}", plot.name))?;
            drawn.push((input, out));
        }
        for (input, out) in drawn {
            write(out, &input, args.out_dir.as_deref())?;
        }
        return Ok(());
    }
    let plot = registry::find(&args.name)
        .ok_or_else(|| format!("unknown figure {:?}; one of: all, {}", args.name, registry::names()))?;
    let out = (plot.render)(&args.input)?;
    write(out, &args.input, args.out_dir.as_deref())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("plot_results: {msg}");
            ExitCode::FAILURE
        }
    }
}
