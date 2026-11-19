//! Forecast time series with TimesFM 3 on the GPU, with no Python anywhere:
//! series in as CSV, quantile forecasts out as CSV.
//!
//! ```text
//! timesfm3_forecast --model DIR --horizon H [--input FILE] [--output FILE]
//!     [--batch-size N] [--znorm] [--symmetric] [--positive] [--no-sort]
//! ```
//!
//! `DIR` holds `config.json` and `model.safetensors` of
//! `google/timesfm-3.0-pytorch` (docs/timesfm3.md says how to fetch them with
//! curl and check them; the weights are under the TimesFM Non-Commercial
//! License v1.0).
//!
//! Input (default stdin) is long-format CSV: a header naming two columns,
//! `id` and `value` (either order), then one row per point in time order. An
//! empty value or `NaN` is a missing point; quoting is not supported. Points
//! of one id need not be adjacent; series are forecast in the order their ids
//! first appear. Output (default stdout) is one row per series and step
//! `1..=H`: `id,step,point,q0.1,...`, where `point` is the median quantile
//! (upstream's `forecast`) and the `q` columns are the checkpoint's
//! quantiles.
//!
//! The options are upstream's `TimesFM3Forecaster.predict_batch()` ones:
//! `--batch-size` is `per_core_batch_size` (default 4; it changes the
//! numbers, since a group shares one context), `--znorm` `use_znorm`,
//! `--symmetric` `use_symmetric_averaging`, `--positive` `make_positive`,
//! `--no-sort` `sort_quantiles=False`. One target variate per series;
//! several variates and covariates are in the library
//! (`tessl::timesfm3_forecast`).

mod common;

use std::io::{Read, Write};
use std::path::PathBuf;

use tessl::safetensors::SafeTensors;
use tessl::timesfm3::{Timesfm3Config, Timesfm3Model};
use tessl::timesfm3_forecast::{PredictOptions, Rows, Series, SeriesForecast};
use tessl::GpuRuntime;

use common::parse_count;

/// Input larger than this is refused before it is parsed.
const MAX_INPUT_BYTES: usize = 1 << 30;

const USAGE: &str = "usage: timesfm3_forecast --model DIR --horizon H [--input FILE] [--output FILE] \
                     [--batch-size N] [--znorm] [--symmetric] [--positive] [--no-sort]";

#[derive(Debug, PartialEq)]
struct Args {
    model: PathBuf,
    horizon: usize,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    opts: PredictOptions,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let (mut model, mut horizon, mut input, mut output) = (None, None, None, None);
    let mut opts = PredictOptions::default();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))
        };
        let once = |slot: &Option<String>, name: &str| {
            if slot.is_some() {
                Err(format!("{name} given twice"))
            } else {
                Ok(())
            }
        };
        match a.as_str() {
            "--model" => {
                once(&model, a)?;
                model = Some(value(a)?);
            }
            "--horizon" => {
                once(&horizon, a)?;
                horizon = Some(value(a)?);
            }
            "--input" => {
                once(&input, a)?;
                input = Some(value(a)?);
            }
            "--output" => {
                once(&output, a)?;
                output = Some(value(a)?);
            }
            "--batch-size" => opts.per_core_batch_size = parse_count(a, &value(a)?, 1)?,
            "--znorm" => opts.use_znorm = true,
            "--symmetric" => opts.use_symmetric_averaging = true,
            "--positive" => opts.make_positive = true,
            "--no-sort" => opts.sort_quantiles = false,
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    let model = PathBuf::from(model.ok_or_else(|| format!("--model is required\n{USAGE}"))?);
    let horizon = parse_count(
        "--horizon",
        &horizon.ok_or_else(|| format!("--horizon is required\n{USAGE}"))?,
        1,
    )?;
    Ok(Args {
        model,
        horizon,
        input: input.map(PathBuf::from),
        output: output.map(PathBuf::from),
        opts,
    })
}

/// Long-format CSV to series, in the order ids first appear.
fn parse_csv(text: &str) -> Result<(Vec<String>, Vec<Vec<f32>>), String> {
    if text.contains('"') {
        return Err("input: quoted CSV fields are not supported".into());
    }
    let mut lines = text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty());
    let (_, header) = lines
        .next()
        .ok_or("input: empty, expected a header naming id and value")?;
    let cols: Vec<&str> = header.split(',').map(str::trim).collect();
    let (id_col, value_col) = match cols.as_slice() {
        ["id", "value"] => (0, 1),
        ["value", "id"] => (1, 0),
        _ => {
            return Err(format!(
                "input: header must name two columns, id and value; got {header:?}"
            ))
        }
    };
    let mut ids: Vec<String> = Vec::new();
    let mut series: Vec<Vec<f32>> = Vec::new();
    let mut index = std::collections::HashMap::new();
    for (n, line) in lines {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() != 2 {
            return Err(format!("input line {}: expected 2 fields, got {}", n + 1, fields.len()));
        }
        let id = fields[id_col];
        if id.is_empty() {
            return Err(format!("input line {}: empty id", n + 1));
        }
        let raw = fields[value_col];
        let v = if raw.is_empty() || raw.eq_ignore_ascii_case("nan") {
            f32::NAN
        } else {
            let v: f32 = raw
                .parse()
                .map_err(|_| format!("input line {}: {raw:?} is not a number", n + 1))?;
            if !v.is_finite() {
                return Err(format!("input line {}: {raw:?} is not finite in f32", n + 1));
            }
            v
        };
        let k = *index.entry(id.to_string()).or_insert_with(|| {
            ids.push(id.to_string());
            series.push(Vec::new());
            series.len() - 1
        });
        series[k].push(v);
    }
    if ids.is_empty() {
        return Err("input: no rows after the header".into());
    }
    Ok((ids, series))
}

fn write_csv(ids: &[String], forecasts: &[SeriesForecast], levels: &[f64]) -> String {
    let mut out = String::from("id,step,point");
    for l in levels {
        out.push_str(&format!(",q{l}"));
    }
    out.push('\n');
    for (id, f) in ids.iter().zip(forecasts) {
        let point = f.point();
        for t in 0..f.horizon {
            out.push_str(&format!("{id},{},{}", t + 1, point[t]));
            for q in &f.values[t * f.quantiles..(t + 1) * f.quantiles] {
                out.push_str(&format!(",{q}"));
            }
            out.push('\n');
        }
    }
    out
}

fn read_input(path: &Option<PathBuf>) -> Result<String, String> {
    let mut buf = Vec::new();
    let n = match path {
        Some(p) => std::fs::File::open(p)
            .map_err(|e| format!("{}: {e}", p.display()))?
            .take(MAX_INPUT_BYTES as u64 + 1)
            .read_to_end(&mut buf),
        None => std::io::stdin().take(MAX_INPUT_BYTES as u64 + 1).read_to_end(&mut buf),
    }
    .map_err(|e| format!("input: {e}"))?;
    if n > MAX_INPUT_BYTES {
        return Err(format!("input: larger than {MAX_INPUT_BYTES} bytes"));
    }
    String::from_utf8(buf).map_err(|_| "input: not UTF-8".to_string())
}

fn run(args: &Args) -> Result<(), String> {
    let (ids, values) = parse_csv(&read_input(&args.input)?)?;
    let cfg = Timesfm3Config::from_config_file(&args.model.join("config.json"))?;
    let levels = cfg.quantile_levels.clone();
    let rt = GpuRuntime::new()?;
    let st = SafeTensors::open(&args.model.join("model.safetensors")).map_err(|e| e.to_string())?;
    let model = Timesfm3Model::load(&rt, &st, cfg)?;
    drop(st);
    let series: Vec<Series> = values
        .into_iter()
        .map(|v| Series {
            target: Rows::one(v),
            ..Series::default()
        })
        .collect();
    let forecasts = model.predict_batch(&series, args.horizon, &args.opts)?;
    let text = write_csv(&ids, &forecasts, &levels);
    match &args.output {
        Some(p) => std::fs::write(p, text).map_err(|e| format!("{}: {e}", p.display())),
        None => std::io::stdout()
            .write_all(text.as_bytes())
            .map_err(|e| format!("stdout: {e}")),
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let result = parse_args(&argv).and_then(|a| run(&a));
    if let Err(e) = result {
        eprintln!("timesfm3_forecast: {e}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn arguments_map_to_upstreams_options() {
        let a = parse_args(&args(
            "--model m --horizon 24 --batch-size 3 --znorm --symmetric --positive --no-sort",
        ))
        .unwrap();
        assert_eq!(a.model, PathBuf::from("m"));
        assert_eq!(a.horizon, 24);
        assert_eq!((a.input, a.output), (None, None));
        let want = PredictOptions {
            per_core_batch_size: 3,
            use_znorm: true,
            use_symmetric_averaging: true,
            make_positive: true,
            sort_quantiles: false,
            ..PredictOptions::default()
        };
        assert_eq!(a.opts, want);
        assert_eq!(
            parse_args(&args("--horizon 1 --model m")).unwrap().opts,
            PredictOptions::default()
        );
    }

    #[test]
    fn bad_arguments_are_refused_by_name() {
        for (cmd, want) in [
            ("--horizon 4", "--model is required"),
            ("--model m", "--horizon is required"),
            ("--model m --horizon 0", "below the minimum"),
            ("--model m --horizon -3", "not a non-negative integer"),
            ("--model m --horizon 4 --batch-size 0", "below the minimum"),
            ("--model m --model n --horizon 4", "given twice"),
            ("--model m --horizon 4 --frobnicate", "unknown argument"),
            ("--model", "needs a value"),
        ] {
            let e = parse_args(&args(cmd))
                .err()
                .unwrap_or_else(|| panic!("{cmd}: accepted"));
            assert!(e.contains(want), "{cmd}: expected {want:?} in {e:?}");
        }
    }

    #[test]
    fn csv_series_follow_first_appearance_and_keep_gaps() {
        let (ids, s) = parse_csv("value,id\n1,b\n2,a\n,b\nNaN,a\n3.5,b\n\n").unwrap();
        assert_eq!(ids, ["b", "a"]);
        assert_eq!(s[0][0], 1.0);
        assert!(s[0][1].is_nan());
        assert_eq!(s[0][2], 3.5);
        assert_eq!(s[1][0], 2.0);
        assert!(s[1][1].is_nan());
    }

    #[test]
    fn bad_csv_is_refused_with_its_line() {
        for (text, want) in [
            ("", "empty"),
            ("id,value\n", "no rows"),
            ("id,val\na,1\n", "header must name"),
            ("id,value\na,1,2\n", "line 2: expected 2 fields"),
            ("id,value\n,1\n", "line 2: empty id"),
            ("id,value\na,x\n", "line 2: \"x\" is not a number"),
            ("id,value\na,1\na,inf\n", "line 3: \"inf\" is not finite"),
            ("id,value\na,1e40\n", "not finite"),
            ("id,value\n\"a\",1\n", "quoted"),
        ] {
            let e = parse_csv(text).err().unwrap_or_else(|| panic!("{text:?}: accepted"));
            assert!(e.contains(want), "{text:?}: expected {want:?} in {e:?}");
        }
    }

    #[test]
    fn output_has_one_row_per_step_with_the_median_as_point() {
        let f = SeriesForecast {
            variates: 1,
            horizon: 2,
            quantiles: 3,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let text = write_csv(&["s".to_string()], &[f], &[0.25, 0.5, 0.75]);
        assert_eq!(text, "id,step,point,q0.25,q0.5,q0.75\ns,1,2,1,2,3\ns,2,5,4,5,6\n");
    }
}
