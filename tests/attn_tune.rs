//! `attn_tune` (`src/bin/common/tune.rs`): the attention knob sweeps.
//!
//! Device-free. The bin runs `bench_flash_attn` once per knob value; here that
//! binary is a shell script that answers from files named by the knob values it
//! was handed and records the environment of every call. So what is checked is
//! everything the tuner does around the measurement: which knobs it sets and
//! clears, the order it visits values in, how it reads the rows back, the table
//! and the JSON it prints, and what it refuses.
//!
//! The expected tables and JSON below are not derived from the code under test:
//! they are what the retired `bench/attn_tune.py` printed for the same inputs
//! (the bytes were compared when the port landed, including stdout, stderr, the
//! exit status, the `--out` file and each child's environment).
//!
//! The first group is the knob-table contract of the retired
//! `bench/test_parity_harness.py::tune_knob_contract`.

#[path = "../src/json.rs"]
#[allow(dead_code)]
mod json;
#[path = "../src/bin/common/proc.rs"]
mod proc;
#[path = "../src/bin/common/pyjson.rs"]
mod pyjson;
#[path = "../src/bin/common/tune.rs"]
mod tune;

use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Command, Output};

const FLASH_ATTN_SOURCE: &str = include_str!("../src/bin/bench_flash_attn.rs");

/// The stand-in `bench_flash_attn`.
const FAKE_BENCH: &str = "#!/bin/sh
d=\"$FAKE_DIR\"
key=\"${BENCH_ATTN_ROWS_R}${BENCH_ATTN_ROWS_SGT}${BENCH_ATTN_DECODE_CHUNK}${BENCH_ATTN_DECODE_R}${BENCH_ATTN_DECODE_SGS}${BENCH_ATTN_REDUCE_W}_b${BENCH_ATTN_BATCHED}\"
echo \"$key\" >> \"$d/calls\"
n=$(grep -c -x -- \"$key\" \"$d/calls\")
env | sort > \"$d/env.$key.$n\"
if [ -f \"$d/fail.$key\" ]; then
    echo \"fake bench_flash_attn failed\" >&2
    exit 3
fi
if [ -f \"$d/out.$key.$n\" ]; then
    cat \"$d/out.$key.$n\"
else
    cat \"$d/out.$key\"
fi
";

// ------------------------------------------------------------------ fixtures

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("tessl-attn-tune-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn knob(name: &str) -> &'static tune::Knob {
    tune::knob(name).unwrap_or_else(|| panic!("no knob {name}"))
}

/// The key the stand-in files its answers under.
fn key(k: &tune::Knob, value: &str, batched: usize) -> String {
    if k.name == "batched" {
        format!("_b{value}")
    } else {
        format!("{value}_b{batched}")
    }
}

/// `bench_flash_attn`'s JSON for `lane`, with a second lane that has better
/// numbers and must be ignored, and fields the tuner does not read.
fn rows(lane: &str, cfgs: &[&str], ms: &[f64]) -> String {
    let mut out: Vec<String> = cfgs
        .iter()
        .zip(ms)
        .map(|(c, m)| {
            format!("{{\"cfg\":\"{c}\",\"runtime\":\"{lane}\",\"median_ms\":{m},\"iters\":20,\"extra\":[1,2]}}")
        })
        .collect();
    out.extend(
        cfgs.iter()
            .map(|c| format!("{{\"cfg\":\"{c}\",\"runtime\":\"other\",\"median_ms\":0.001}}")),
    );
    format!("[{}]", out.join(","))
}

/// ms per (config, value) for the six-wide knobs: the winner varies per config,
/// one config ties, and one has a gain that needs rounding.
const BASE: [[f64; 3]; 7] = [
    [2.0, 1.0, 3.0],
    [4.0, 5.0, 4.5],
    [1.5, 1.5, 1.5],
    [0.25, 0.5, 0.125],
    [10.0, 8.0, 9.0],
    [0.3333333, 0.30000004, 0.3000001],
    [7.0, 7.5, 6.9],
];

struct Fake {
    dir: PathBuf,
    bin: PathBuf,
}

impl Fake {
    fn new(tag: &str) -> Fake {
        let dir = scratch(tag);
        let bin = dir.join("bench_flash_attn");
        std::fs::write(&bin, FAKE_BENCH).unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        Fake { dir, bin }
    }

    fn put(&self, name: &str, body: &str) {
        std::fs::write(self.dir.join(name), body).unwrap();
    }

    /// Answer every value of `k`: from `table(config, value)`, and, when
    /// `outlier`, answer a value's second call with 100x that, so a median over
    /// rounds is told apart from a mean.
    fn populate(
        &self,
        k: &tune::Knob,
        batched: usize,
        outlier: bool,
        drop: Option<&str>,
        table: &dyn Fn(usize, usize) -> f64,
    ) {
        let cfgs: Vec<&str> = k.cfgs.iter().copied().filter(|c| Some(*c) != drop).collect();
        for (vi, v) in k.values.iter().enumerate() {
            let ms: Vec<f64> = (0..cfgs.len()).map(|ci| table(ci, vi)).collect();
            let name = key(k, v, batched);
            self.put(&format!("out.{name}"), &rows(k.lane, &cfgs, &ms));
            if outlier {
                let big: Vec<f64> = ms.iter().map(|x| 100.0 * x).collect();
                self.put(&format!("out.{name}.2"), &rows(k.lane, &cfgs, &big));
            }
        }
    }

    fn base(&self, k: &tune::Knob, batched: usize, outlier: bool) {
        self.populate(k, batched, outlier, None, &|ci, vi| BASE[ci % BASE.len()][vi % 3]);
    }

    fn run(&self, args: &[&str], env: &[(&str, &str)]) -> Output {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_attn_tune"));
        cmd.arg("--binary").arg(&self.bin).args(args).env("FAKE_DIR", &self.dir);
        // The caller's own BENCH_* settings are not the test's business.
        for (name, _) in std::env::vars() {
            if name.starts_with("BENCH_") {
                cmd.env_remove(name);
            }
        }
        for (k, v) in env {
            cmd.env(k, v);
        }
        cmd.output().unwrap()
    }

    fn calls(&self) -> Vec<String> {
        std::fs::read_to_string(self.dir.join("calls"))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    /// The `BENCH_*` environment of the `n`th call filed under `key`.
    fn child_env(&self, key: &str, n: usize) -> BTreeMap<String, String> {
        std::fs::read_to_string(self.dir.join(format!("env.{key}.{n}")))
            .unwrap_or_else(|e| panic!("no env.{key}.{n}: {e}"))
            .lines()
            .filter(|l| l.starts_with("BENCH_"))
            .map(|l| {
                let (k, v) = l.split_once('=').unwrap();
                (k.to_string(), v.to_string())
            })
            .collect()
    }
}

fn text(b: &[u8]) -> String {
    String::from_utf8(b.to_vec()).unwrap()
}

fn code(o: &Output) -> i32 {
    o.status.code().expect("killed by a signal")
}

// ------------------------------------------------- the knob-table contract

/// Python `attn_tune.KNOB_ENV == {env for every knob}`.
#[test]
fn knob_env_covers_every_knob() {
    let declared: BTreeSet<&str> = tune::KNOBS.iter().map(|k| k.env).collect();
    let listed: BTreeSet<&str> = tune::knob_env().into_iter().collect();
    assert_eq!(declared, listed);
    assert_eq!(tune::knob_env().len(), listed.len(), "knob_env lists a variable twice");
    assert_eq!(declared.len(), 7, "one variable per knob");
}

/// The knob names are unique and are exactly what `--knob` accepts.
#[test]
fn knob_names_are_unique_and_sorted() {
    let names = tune::knob_names();
    assert_eq!(
        names,
        [
            "batched",
            "decode",
            "decode-r",
            "decode-sgs",
            "reduce-w",
            "rows",
            "rows-g"
        ]
    );
    for n in &names {
        assert_eq!(tune::knob(n).unwrap().name, *n);
    }
    assert!(tune::knob("nope").is_none());
}

/// Python: every knob config is a config the benchmark has, its lane is one the
/// binary emits, and no value is listed twice. The configs and lanes are read
/// from `bench_flash_attn`'s own source, the file the tuner drives.
#[test]
fn knobs_name_real_configs_and_lanes() {
    let labels: BTreeSet<&str> = FLASH_ATTN_SOURCE
        .lines()
        .filter_map(|l| l.trim().strip_prefix("label: \""))
        .filter_map(|r| r.split('"').next())
        .collect();
    let lanes: BTreeSet<&str> = FLASH_ATTN_SOURCE
        .lines()
        .filter_map(|l| l.split_once("=> \"tessl").map(|(_, r)| r))
        .filter_map(|r| r.split('"').next())
        .map(|suffix| match suffix {
            "" => "tessl",
            "-tiled" => "tessl-tiled",
            "-decode" => "tessl-decode",
            "-rows" => "tessl-rows",
            other => panic!("unexpected lane suffix {other:?}"),
        })
        .collect();
    assert_eq!(lanes.len(), 4, "lanes found in bench_flash_attn.rs: {lanes:?}");
    assert!(labels.len() >= 14, "configs found in bench_flash_attn.rs: {labels:?}");

    for k in tune::KNOBS {
        let bad: Vec<&&str> = k.cfgs.iter().filter(|c| !labels.contains(**c)).collect();
        assert!(bad.is_empty(), "knob {}: unknown configs {bad:?}", k.name);
        assert!(
            lanes.contains(k.lane),
            "knob {}: lane {:?} is not one the binary emits",
            k.name,
            k.lane
        );
        let distinct: BTreeSet<&&str> = k.values.iter().collect();
        assert_eq!(
            distinct.len(),
            k.values.len(),
            "knob {}: duplicate values {:?}",
            k.name,
            k.values
        );
        let cfgs: BTreeSet<&&str> = k.cfgs.iter().collect();
        assert_eq!(cfgs.len(), k.cfgs.len(), "knob {}: duplicate configs", k.name);
        assert!(k.cfgs.iter().all(|c| c.contains("prefill") || c.contains("decode")));
    }
}

/// The decomposition divides batched by solo, so the values and their order are
/// part of the contract.
#[test]
fn batched_knob_is_solo_then_amortised() {
    assert_eq!(knob("batched").values, ["1", "32"]);
}

/// `render` refuses a `batched` table whose values are not `('1', '32')`.
#[test]
fn render_refuses_reordered_batched_values() {
    let bad = tune::Knob {
        name: "batched",
        env: "BENCH_ATTN_BATCHED",
        values: &["32", "1"],
        lane: "tessl",
        cfgs: &["swa128_decode_1k"],
    };
    let per: tune::PerRound = vec![vec![vec![1.0]], vec![vec![2.0]]];
    let err = tune::render(&bad, 1, 1, 1, &per)
        .err()
        .expect("accepted reordered values");
    assert!(err.contains("expects values ('1', '32'), got ['32', '1']"), "{err}");
}

/// The environment edits for one run: the knobs not being swept are removed, the
/// batching is pinned, and the swept knob is set last so `--knob batched` can
/// override the pin.
#[test]
fn env_edits_clear_the_other_knobs_and_set_the_swept_one_last() {
    let edits = tune::env_edits("BENCH_ATTN_ROWS_R", "8", &["a", "b"], 7, 2, 1);
    let last = edits.last().unwrap();
    assert_eq!(last, &("BENCH_ATTN_ROWS_R".to_string(), Some("8".to_string())));
    let removed: BTreeSet<&str> = edits
        .iter()
        .filter(|(_, v)| v.is_none())
        .map(|(k, _)| k.as_str())
        .collect();
    let expected: BTreeSet<&str> = tune::knob_env().into_iter().collect();
    assert_eq!(removed, expected, "every knob variable is removed first");
    let set: BTreeMap<&str, &str> = edits
        .iter()
        .filter_map(|(k, v)| v.as_deref().map(|v| (k.as_str(), v)))
        .collect();
    assert_eq!(set["BENCH_ITERS"], "7");
    assert_eq!(set["BENCH_WARMUP"], "2");
    assert_eq!(set["BENCH_ATTN_CFGS"], "a,b");
    assert_eq!(set["BENCH_ATTN_BATCHED"], "1");
    // The batched sweep overrides the pin because the knob is applied after it.
    let b = tune::env_edits("BENCH_ATTN_BATCHED", "32", &["a"], 1, 0, 1);
    let order: Vec<Option<&str>> = b
        .iter()
        .filter(|(k, _)| k == "BENCH_ATTN_BATCHED")
        .map(|(_, v)| v.as_deref())
        .collect();
    assert_eq!(
        order,
        [None, Some("1"), Some("32")],
        "cleared, pinned, then the swept value wins"
    );
}

// -------------------------------------------- what the child process receives

/// Python: sweeping one knob clears the others and pins the batching, so a knob
/// or a batching left in the caller's shell is not measured as part of the value.
#[test]
fn a_sweep_clears_stray_knobs_and_pins_the_batching() {
    let fake = Fake::new("env");
    let k = knob("rows");
    fake.base(k, 1, false);
    let stray = [
        ("BENCH_ATTN_DECODE_CHUNK", "256"),
        ("BENCH_ATTN_BATCHED", "32"),
        ("BENCH_ATTN_REDUCE_W", "32"),
        ("BENCH_ATTN_ROWS_SGT", "8"),
        ("BENCH_ATTN_DECODE_SGS", "all"),
        ("BENCH_ATTN_DECODE_R", "8"),
    ];
    let o = fake.run(
        &["--knob", "rows", "--rounds", "1", "--iters", "7", "--warmup", "2"],
        &stray,
    );
    assert_eq!(code(&o), 0, "{}", text(&o.stderr));
    for v in k.values {
        let seen = fake.child_env(&key(k, v, 1), 1);
        let want: BTreeMap<String, String> = [
            ("BENCH_ATTN_BATCHED", "1"),
            ("BENCH_ATTN_CFGS", &k.cfgs.join(",")),
            ("BENCH_ATTN_ROWS_R", v),
            ("BENCH_ITERS", "7"),
            ("BENCH_WARMUP", "2"),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
        assert_eq!(seen, want, "environment of the {v} run");
    }
}

/// `--batched N` reaches every arm of a sweep and the header names it.
#[test]
fn batching_reaches_every_arm() {
    let fake = Fake::new("batched-arm");
    let k = knob("decode");
    fake.base(k, 32, false);
    let o = fake.run(&["--knob", "decode", "--rounds", "1", "--batched", "32"], &[]);
    assert_eq!(code(&o), 0, "{}", text(&o.stderr));
    for v in k.values {
        assert_eq!(fake.child_env(&key(k, v, 32), 1)["BENCH_ATTN_BATCHED"], "32");
    }
    let out = text(&o.stdout);
    assert!(
        out.contains(
            "BENCH_ATTN_DECODE_CHUNK  (median ms over 1 interleaved rounds, kernel only, 32 launches per submit)\n"
        ),
        "{out}"
    );
    assert_eq!(text(&o.stderr), "round 1/1 done\n");
}

/// The swept knob is set to each value, in the order round-robin over the values
/// rather than all rounds of one and then the next, so thermal drift cannot
/// favour whichever was measured first.
#[test]
fn values_are_visited_round_robin() {
    let fake = Fake::new("order");
    let k = knob("rows");
    fake.base(k, 1, true);
    let o = fake.run(&["--knob", "rows", "--rounds", "3"], &[]);
    assert_eq!(code(&o), 0, "{}", text(&o.stderr));
    let expect: Vec<String> = (0..3)
        .flat_map(|_| ["8_b1", "16_b1", "32_b1"])
        .map(str::to_string)
        .collect();
    assert_eq!(fake.calls(), expect);
}

/// Defaults: three rounds of 20 iterations after 5 warmups, no output file.
#[test]
fn defaults_are_three_rounds_of_twenty_after_five() {
    let fake = Fake::new("defaults");
    let k = knob("rows");
    fake.base(k, 1, false);
    let o = fake.run(&["--knob", "rows"], &[]);
    assert_eq!(code(&o), 0, "{}", text(&o.stderr));
    let env = fake.child_env("8_b1", 1);
    assert_eq!((env["BENCH_ITERS"].as_str(), env["BENCH_WARMUP"].as_str()), ("20", "5"));
    assert_eq!(fake.calls().len(), 9);
    assert_eq!(text(&o.stderr), "round 1/3 done\nround 2/3 done\nround 3/3 done\n");
}

// ------------------------------------------------------ the report

const ROWS_TABLE: &str = "
BENCH_ATTN_ROWS_R  (median ms over 3 interleaved rounds, wall clock, submit-and-wait per call)
  config                           8        16        32    winner    gain
  swa128_prefill_512           2.000     1.000     3.000        16   3.00x
  swa128_prefill_2048          4.000     5.000     4.500         8   1.25x
  swa128_prefill_4096          1.500     1.500     1.500         8   1.00x
  swa256_prefill_2048          0.250     0.500     0.125        32   4.00x
  global512_prefill_1024      10.000     8.000     9.000        16   1.25x
";

const ROWS_JSON: &str = r#"{
  "knob": "rows",
  "env": "BENCH_ATTN_ROWS_R",
  "values": [
    "8",
    "16",
    "32"
  ],
  "rounds": 3,
  "iters": 7,
  "batched": 1,
  "median_ms": {
    "8": {
      "swa128_prefill_512": 2.0,
      "swa128_prefill_2048": 4.0,
      "swa128_prefill_4096": 1.5,
      "swa256_prefill_2048": 0.25,
      "global512_prefill_1024": 10.0
    },
    "16": {
      "swa128_prefill_512": 1.0,
      "swa128_prefill_2048": 5.0,
      "swa128_prefill_4096": 1.5,
      "swa256_prefill_2048": 0.5,
      "global512_prefill_1024": 8.0
    },
    "32": {
      "swa128_prefill_512": 3.0,
      "swa128_prefill_2048": 4.5,
      "swa128_prefill_4096": 1.5,
      "swa256_prefill_2048": 0.125,
      "global512_prefill_1024": 9.0
    }
  },
  "winner": {
    "swa128_prefill_512": "16",
    "swa128_prefill_2048": "8",
    "swa128_prefill_4096": "8",
    "swa256_prefill_2048": "32",
    "global512_prefill_1024": "16"
  }
}"#;

/// The table, the winner and gain per config (the first of equal minima wins,
/// a tie reads 1.00x), the median over rounds (round two is 100x, so a mean
/// would show), the other lane ignored.
#[test]
fn the_table_names_a_winner_per_config() {
    let fake = Fake::new("table");
    fake.base(knob("rows"), 1, true);
    let o = fake.run(
        &["--knob", "rows", "--rounds", "3", "--iters", "7", "--warmup", "2"],
        &[],
    );
    assert_eq!(code(&o), 0, "{}", text(&o.stderr));
    assert_eq!(text(&o.stdout), ROWS_TABLE);
}

/// `--out` writes the document with `json.dumps(indent=2)`'s bytes and no
/// trailing newline, and says so on stderr; the document is the knob, its
/// variable, the values, the rounds, the iterations, the batching, the medians
/// and the winners.
#[test]
fn out_writes_the_document_byte_for_byte() {
    let fake = Fake::new("out");
    fake.base(knob("rows"), 1, true);
    let out = fake.dir.join("result.json");
    let o = fake.run(
        &[
            "--knob",
            "rows",
            "--rounds",
            "3",
            "--iters",
            "7",
            "--warmup",
            "2",
            "--out",
            out.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code(&o), 0, "{}", text(&o.stderr));
    assert_eq!(std::fs::read_to_string(&out).unwrap(), ROWS_JSON);
    assert_eq!(text(&o.stdout), ROWS_TABLE);
    assert_eq!(
        text(&o.stderr),
        format!(
            "round 1/3 done\nround 2/3 done\nround 3/3 done\n\nwrote {}\n",
            out.display()
        )
    );
}

/// Non-numeric values (`one`, `group`, `all`) and the sixth config of
/// `decode-sgs` go through the same table.
#[test]
fn a_knob_with_named_values_and_six_configs() {
    let fake = Fake::new("sgs");
    let k = knob("decode-sgs");
    assert_eq!(k.cfgs.len(), 6);
    fake.base(k, 1, false);
    let o = fake.run(&["--knob", "decode-sgs", "--rounds", "1"], &[]);
    assert_eq!(code(&o), 0, "{}", text(&o.stderr));
    let out = text(&o.stdout);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(
        lines[1],
        "BENCH_ATTN_DECODE_SGS  (median ms over 1 interleaved rounds, wall clock, submit-and-wait per call)"
    );
    assert_eq!(
        lines[2],
        "  config                         one     group       all    winner    gain"
    );
    assert_eq!(lines.len(), 3 + 6);
    // global512_decode_4k_mha is the sixth config: base row 5 is [0.3333333, 0.30000004, 0.3000001].
    assert_eq!(
        lines[8],
        "  global512_decode_4k_mha      0.333     0.300     0.300     group   1.11x"
    );
}

const BATCHED_TABLE: &str = "
BENCH_ATTN_BATCHED  (median ms over 2 interleaved rounds, wall clock, submit-and-wait per call)
  config                     solo ms  batched ms  submit ms  kernel share
  swa128_decode_1k           50.5000     12.6250    37.8750         25.0%
  swa128_decode_4k          101.0000     75.7500    25.2500         75.0%
  swa256_decode_4k           25.2500     25.2500     0.0000        100.0%
  global512_decode_4k       202.0000     50.5000   151.5000         25.0%
  swa128_prefill_2048       151.5000    151.5000     0.0000        100.0%
  swa256_prefill_2048       303.0000     15.1500   287.8500          5.0%
  global512_prefill_1024    404.0000    454.5000   -50.5000        112.5%
";

/// `--knob batched` is a decomposition: solo against 32 launches per submit, the
/// share of the wall clock that is kernel, and no winner.
#[test]
fn the_batched_knob_reports_a_kernel_share_not_a_winner() {
    let fake = Fake::new("share");
    let k = knob("batched");
    let solo = [1.0, 2.0, 0.5, 4.0, 3.0, 6.0, 8.0];
    let amortised = [0.25, 1.5, 0.5, 1.0, 3.0, 0.3, 9.0];
    fake.populate(k, 1, true, None, &|ci, vi| {
        if vi == 0 {
            solo[ci]
        } else {
            amortised[ci]
        }
    });
    let out = fake.dir.join("share.json");
    let o = fake.run(
        &["--knob", "batched", "--rounds", "2", "--out", out.to_str().unwrap()],
        &[],
    );
    assert_eq!(code(&o), 0, "{}", text(&o.stderr));
    assert_eq!(text(&o.stdout), BATCHED_TABLE);
    let doc = std::fs::read_to_string(&out).unwrap();
    assert!(doc.contains("\"winner\": null,\n  \"kernel_share\": {\n"), "{doc}");
    assert!(doc.contains("\"swa128_decode_1k\": 0.25,\n"), "{doc}");
    assert!(doc.contains("\"global512_prefill_1024\": 1.125\n  }\n}"), "{doc}");
    assert!(!doc.ends_with('\n'));
    assert_eq!(
        fake.calls(),
        ["_b1", "_b32", "_b1", "_b32"],
        "the swept batching is set per arm, not pinned"
    );
}

// ------------------------------------------------------ what it refuses

fn refused(o: &Output) -> String {
    assert_eq!(code(o), 1, "stdout: {}", text(&o.stdout));
    assert!(
        o.stdout.is_empty(),
        "a refused sweep printed a report: {}",
        text(&o.stdout)
    );
    text(&o.stderr)
}

/// Python: a lane that lacks a configuration is a partial sweep, and no winner
/// is picked from it.
#[test]
fn a_partial_sweep_is_refused() {
    let fake = Fake::new("partial");
    let k = knob("rows");
    fake.populate(k, 1, false, Some("swa256_prefill_2048"), &|ci, vi| BASE[ci % 7][vi]);
    let o = fake.run(&["--knob", "rows", "--rounds", "1"], &[]);
    assert_eq!(
        refused(&o),
        "BENCH_ATTN_ROWS_R=8: lane 'tessl-rows' absent for ['swa256_prefill_2048']. Refusing to pick a winner from a partial sweep.\n"
    );
    assert_eq!(fake.calls(), ["8_b1"], "stopped at the first bad run");
}

/// A run that exits nonzero names the value and carries the child's stderr.
#[test]
fn a_failed_run_is_refused_with_its_stderr() {
    let fake = Fake::new("fail");
    let k = knob("rows");
    fake.base(k, 1, false);
    fake.put("fail.16_b1", "");
    let o = fake.run(&["--knob", "rows", "--rounds", "1"], &[]);
    assert_eq!(
        refused(&o),
        "BENCH_ATTN_ROWS_R=16: exited 3\nfake bench_flash_attn failed\n"
    );
    assert_eq!(fake.calls(), ["8_b1", "16_b1"]);
}

/// A run that prints something that is not the rows is refused, with a piece of
/// what it printed, instead of reading as an empty lane.
#[test]
fn output_that_is_not_rows_is_refused() {
    let fake = Fake::new("garbage");
    fake.put("out.8_b1", "this is not json");
    let o = fake.run(&["--knob", "rows", "--rounds", "1"], &[]);
    let err = refused(&o);
    assert!(err.starts_with("BENCH_ATTN_ROWS_R=8: "), "{err}");
    assert!(err.contains("this is not json"), "{err}");

    fake.put("out.8_b1", "{\"cfg\": \"x\"}");
    let err = refused(&fake.run(&["--knob", "rows", "--rounds", "1"], &[]));
    assert!(err.contains("not an array of rows"), "{err}");

    fake.put(
        "out.8_b1",
        "[{\"cfg\":\"swa128_prefill_512\",\"runtime\":\"tessl-rows\"}]",
    );
    let err = refused(&fake.run(&["--knob", "rows", "--rounds", "1"], &[]));
    assert!(
        err.contains("row 0 lacks a string cfg, a string runtime or a numeric median_ms"),
        "{err}"
    );
}

/// Python: `--knob batched` sweeps launches-per-submit, so `--batched` would fix
/// the very thing being swept.
#[test]
fn knob_batched_with_batched_is_refused() {
    let fake = Fake::new("batched-twice");
    let o = fake.run(&["--knob", "batched", "--batched", "32"], &[]);
    assert_eq!(
        refused(&o),
        "--knob batched sweeps launches-per-submit; --batched would fix the very thing being swept. Drop one of them.\n"
    );
    assert!(fake.calls().is_empty(), "ran the binary before refusing");
}

#[test]
fn counts_that_cannot_run_are_refused_before_anything_runs() {
    let fake = Fake::new("counts");
    assert_eq!(
        refused(&fake.run(&["--knob", "rows", "--rounds", "0"], &[])),
        "--rounds must be >= 1, got 0\n"
    );
    assert_eq!(
        refused(&fake.run(&["--knob", "rows", "--rounds", "-3"], &[])),
        "--rounds must be >= 1, got -3\n"
    );
    assert_eq!(
        refused(&fake.run(&["--knob", "rows", "--batched", "0"], &[])),
        "--batched must be >= 1, got 0\n"
    );
    let err = refused(&fake.run(&["--knob", "rows", "--iters", "-1"], &[]));
    assert!(err.contains("--iters must be >= 0, got -1"), "{err}");
    let err = refused(&fake.run(&["--knob", "rows", "--warmup", "-1"], &[]));
    assert!(err.contains("--warmup must be >= 0, got -1"), "{err}");
    assert!(fake.calls().is_empty());
}

/// A binary that is not built is named, with the command that builds it.
#[test]
fn a_missing_binary_is_named() {
    let dir = scratch("nobin");
    let absent = dir.join("absent");
    let o = Command::new(env!("CARGO_BIN_EXE_attn_tune"))
        .args(["--knob", "rows", "--binary"])
        .arg(&absent)
        .output()
        .unwrap();
    assert_eq!(
        refused(&o),
        format!(
            "{} not built. cargo build --release --bin bench_flash_attn\n",
            absent.display()
        )
    );
}

/// Without `--binary` the tuner drives the `bench_flash_attn` that sits next to
/// it, which is what `cargo build --release --bins` leaves in `target/release`.
#[test]
fn the_default_binary_is_the_sibling_bench_flash_attn() {
    let fake = Fake::new("sibling");
    fake.base(knob("rows"), 1, false);
    let tuner = fake.dir.join("attn_tune");
    std::fs::copy(env!("CARGO_BIN_EXE_attn_tune"), &tuner).unwrap();
    let mut cmd = Command::new(&tuner);
    cmd.args(["--knob", "rows", "--rounds", "1"]).env("FAKE_DIR", &fake.dir);
    for (name, _) in std::env::vars() {
        if name.starts_with("BENCH_") {
            cmd.env_remove(name);
        }
    }
    let o = cmd.output().unwrap();
    assert_eq!(code(&o), 0, "{}", text(&o.stderr));
    assert_eq!(fake.calls().len(), 3);
}

// --------------------------------------------------------- command line

fn bare(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_attn_tune"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn the_command_line_is_checked() {
    let o = bare(&[]);
    assert_eq!(code(&o), 2);
    assert!(
        text(&o.stderr).contains("the following arguments are required: --knob"),
        "{}",
        text(&o.stderr)
    );

    let o = bare(&["--knob", "nope"]);
    assert_eq!(code(&o), 2);
    let err = text(&o.stderr);
    assert!(err.contains("invalid choice: 'nope' (choose from 'batched', 'decode', 'decode-r', 'decode-sgs', 'reduce-w', 'rows', 'rows-g')"), "{err}");

    let o = bare(&["--knob", "rows", "--rounds", "many"]);
    assert_eq!(code(&o), 2);
    assert!(
        text(&o.stderr).contains("argument --rounds: invalid int value: \"many\""),
        "{}",
        text(&o.stderr)
    );

    let o = bare(&["--knob", "rows", "--rounds"]);
    assert_eq!(code(&o), 2);
    assert!(
        text(&o.stderr).contains("argument --rounds: expected one argument"),
        "{}",
        text(&o.stderr)
    );

    let o = bare(&["--knob", "rows", "--frobnicate"]);
    assert_eq!(code(&o), 2);
    assert!(
        text(&o.stderr).contains("unrecognized arguments: --frobnicate"),
        "{}",
        text(&o.stderr)
    );

    let o = bare(&["--help"]);
    assert_eq!(code(&o), 0);
    assert!(
        text(&o.stdout).starts_with("usage: attn_tune --knob {batched,decode,"),
        "{}",
        text(&o.stdout)
    );
}

/// `--knob=rows` is the same as `--knob rows`.
#[test]
fn equals_form_is_accepted() {
    let fake = Fake::new("equals");
    fake.base(knob("rows"), 1, false);
    let o = fake.run(&["--knob=rows", "--rounds=1"], &[]);
    assert_eq!(code(&o), 0, "{}", text(&o.stderr));
    assert_eq!(fake.calls().len(), 3);
}
