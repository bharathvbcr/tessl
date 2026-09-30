//! The published GDN fixture corpus: its load path, and its own consistency.
//!
//! `tests/gdn.rs` (K1-K7) will validate Metal kernels against these goldens, so
//! this file exists to establish that the goldens load and agree *before* any
//! kernel is written against them. It deliberately does two things that the
//! kernel tests will not: it never touches the GPU (no `with_gpu`, so it runs on
//! any host), and it compares no kernel to anything. It pins the corpus that the
//! later comparison consumes.
//!
//! **Why an f64 reference at all.** Each case ships two outputs: `y_chunked`
//! (f32, the chunked algorithm) and `y_seq_f64` (f64, the sequential reference).
//! Only the f64 one is an *independent* golden. Validating the chunked kernel
//! against `y_chunked` would be self-consistent and prove nothing, since it is
//! the output of the same algorithm under test. `docs/hardening.md:133` in the
//! sibling repo says it plainly: "f64 sequential reference in Rust is the
//! golden". Until this commit `read_npy` could not load it -- it accepted `<f4`
//! and `<i8` only, and returned `unsupported dtype <f8` for all 13 references
//! (`GAP-TESSL-NPY-READER-REFUSES-F64-REFERENCE`).
//!
//! **Repo rule 9.** A GDN fixture that does not name the `published` rule is
//! invalid: nanolab's *default* is `rule="repo"`, a different operator, so a
//! fixture generated at the default is a silently wrong golden.
//! `published_rule_is_declared_by_every_fixture_and_the_manifest` is the guard.

use std::path::PathBuf;

use tessl::npy::{read_npy, write_npy_f32};

mod common;
use common::gdn::{case_names, fixture_dir, load_case, sequential_f64, GateClamp, Rule};

/// f32 unit roundoff, 2^-24. Same value as `tests/common/mod.rs:161`, restated
/// here because this file deliberately does not pull in the GPU test harness.
const U_F32: f64 = 5.960_464_477_539_063e-8;

/// Relative agreement required between the f32 chunked output and the f64
/// sequential golden, as a multiple of `U_F32`.
///
/// Measured over all 13 published cases: the worst is 1.473e-6 = 24.7 u on
/// `L127c64`, and the longest sequence (`L8191`, 8191 steps) is 6.60e-7 = 11.1
/// u. Error is therefore dominated by chunk-local accumulation, not sequence
/// length, so a bound that grows with L would be both wrong in shape and far too
/// loose to bite -- `gamma(8191, U_F32)` is 4.9e-4, which would pass almost any
/// regression. 64 u leaves 2.6x headroom over the worst observed case while
/// still failing a 3x regression. The corpus is static, so this is a
/// deterministic comparison rather than a flaky one.
const REL_BOUND: f64 = 64.0 * U_F32;

/// Relative agreement required between this crate's f64 reference and the
/// Python-generated f64 golden.
///
/// Measured today: **0.0 on every one of the 13 cases** -- bit-for-bit, including
/// `L8191`'s 8191 sequential steps, so the two implementations agree on
/// summation order and not merely on semantics. The test passes with this
/// constant set to exactly `0.0`.
///
/// It is deliberately *not* set to 0.0. Bit-exactness across a Rust/NumPy
/// boundary holds by a coincidence of accumulation order that a torch or numpy
/// upgrade could change without anything being wrong, and a suite that fails on
/// a dependency bump teaches people to ignore it. 1e-12 is ~1e4 x f64 unit
/// roundoff: loose enough to survive a reassociation, tight enough that the two
/// real defects this pair has already caught still fail it by orders of
/// magnitude -- a wrong rule diverges by 1.4e-1, and a missed alpha clamp by
/// 8.4e-5.
const REF_REL_BOUND: f64 = 1e-12;

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Worst absolute difference, and the magnitude it should be judged against.
fn worst_abs(got: &[f64], want: &[f64]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    let mag = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let err = got.iter().zip(want).fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
    (err, mag)
}

/// The regression test for `GAP-TESSL-NPY-READER-REFUSES-F64-REFERENCE`.
///
/// Against the pre-fix reader every case here fails with
/// `unsupported dtype <f8 in ...`, because the match had arms for `<f4` and
/// `<i8` only.
#[test]
fn the_f64_sequential_reference_loads() {
    for case in case_names() {
        let path = fixture_dir().join(format!("gdn_published_{case}_y_seq_f64.npy"));
        let arr = read_npy(&path).unwrap_or_else(|e| panic!("load {case} reference: {e}"));
        let data = arr
            .f64_slice()
            .unwrap_or_else(|e| panic!("{case} reference is not f64: {e}"));
        assert_eq!(
            data.len(),
            numel(&arr.shape),
            "{case}: payload length must match the header shape {:?}",
            arr.shape
        );
        assert!(
            data.iter().all(|v| v.is_finite()),
            "{case}: reference contains a non-finite value"
        );
    }
}

/// An f64 golden must not be reachable through the f32 accessor.
///
/// The cheap way to make the previous test pass would be to decode `<f8` into
/// `data_f32`, which silently throws away the extra precision that is the entire
/// reason the reference is generated in f64. This fails if anyone does that.
#[test]
fn the_f64_reference_is_not_silently_narrowed_to_f32() {
    let case = &case_names()[0];
    let path = fixture_dir().join(format!("gdn_published_{case}_y_seq_f64.npy"));
    let arr = read_npy(&path).expect("reference loads");

    assert!(arr.data_f32.is_none(), "an <f8 file must not populate data_f32");
    let err = arr
        .f32_slice()
        .expect_err("f32_slice on an f64 array must be refused, not coerced");
    assert!(
        err.contains("float32"),
        "refusal should name the expected dtype, got: {err}"
    );
}

/// Both halves of every case are present. A missing f64 reference would
/// otherwise silently shrink the set the parity test iterates.
#[test]
fn every_case_ships_both_a_chunked_output_and_an_f64_reference() {
    let cases = case_names();
    for case in &cases {
        for suffix in ["y_chunked.npy", "y_seq_f64.npy", "q.npy", "k.npy", "v.npy"] {
            let path = fixture_dir().join(format!("gdn_published_{case}_{suffix}"));
            assert!(path.is_file(), "missing {}", path.display());
        }
    }
    assert_eq!(cases.len(), 13, "expected the 13 published cases; found {cases:?}");
}

/// The corpus is internally consistent: the chunked f32 output agrees with the
/// independent f64 sequential reference.
///
/// This is what makes the goldens trustworthy as an input to K1-K7. If a
/// fixture pair is ever regenerated inconsistently -- a different seed, a
/// different operator rule, a truncated write -- this is what notices.
#[test]
fn the_chunked_output_agrees_with_the_f64_sequential_reference() {
    let mut worst = (String::new(), 0.0f64);

    for case in case_names() {
        let chunked = read_npy(&fixture_dir().join(format!("gdn_published_{case}_y_chunked.npy")))
            .unwrap_or_else(|e| panic!("load {case} chunked: {e}"));
        let reference = read_npy(&fixture_dir().join(format!("gdn_published_{case}_y_seq_f64.npy")))
            .unwrap_or_else(|e| panic!("load {case} reference: {e}"));

        assert_eq!(
            chunked.shape, reference.shape,
            "{case}: chunked and reference disagree on shape"
        );

        let got = chunked.f32_slice().expect("chunked output is f32");
        let want = reference.f64_slice().expect("reference is f64");
        assert_eq!(got.len(), want.len(), "{case}: element count");

        // The denominator is the magnitude of the whole tensor, not of each
        // element: an element near zero cannot be held to a relative bound, and
        // judging it by its own value would demand precision f32 cannot give.
        let mag = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(mag > 0.0, "{case}: reference is all zeros, nothing is tested");
        let bound = REL_BOUND * mag;

        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(g.is_finite(), "{case}[{i}]: chunked output is non-finite: {g}");
            let err = (f64::from(*g) - *w).abs();
            assert!(
                err <= bound,
                "{case}[{i}]: |{g} - {w}| = {err:.3e} exceeds {bound:.3e} \
                 ({:.1} u_f32 against a {:.1} u_f32 budget)",
                err / mag / U_F32,
                REL_BOUND / U_F32,
            );
            if err / mag > worst.1 {
                worst = (case.clone(), err / mag);
            }
        }
    }

    // Carry the measurement, not just the verdict: a corpus that quietly drifted
    // to just inside the bound should be visible in the log rather than silent.
    println!(
        "worst relative disagreement: {:.3e} ({:.1} u_f32) on {}, budget {:.1} u_f32",
        worst.1,
        worst.1 / U_F32,
        worst.0,
        REL_BOUND / U_F32
    );
}

/// Repo rule 9: parity targets name the rule.
#[test]
fn published_rule_is_declared_by_every_fixture_and_the_manifest() {
    let entries: Vec<String> = std::fs::read_dir(fixture_dir())
        .expect("fixture directory must exist")
        .map(|e| e.expect("readable dir entry").file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .collect();
    assert!(!entries.is_empty(), "fixture directory is empty");

    for name in &entries {
        assert!(
            name.contains("published"),
            "rule 9: every GDN fixture must name its operator rule, but {name} does not. \
             nanolab's default is rule=\"repo\", a different operator, so an unnamed \
             fixture is a silently wrong golden."
        );
    }

    let manifest =
        std::fs::read_to_string(fixture_dir().join("gdn_published_MANIFEST.json")).expect("manifest must exist");
    assert!(
        manifest.contains("\"rule\": \"published\""),
        "rule 9: the manifest must declare rule=published"
    );
    assert!(
        !manifest.contains("\"rule\": \"repo\""),
        "rule 9: the manifest declares a repo-rule case; that operator is not the published one"
    );
}

/// Covers the two reader arms this change refactored onto a shared payload
/// helper. `read_npy` had no callers and no tests before this file, so the
/// refactor would otherwise have been unverified.
#[test]
fn f32_and_i64_survive_a_round_trip_through_the_reader() {
    let dir = std::env::temp_dir().join(format!("tessl-npy-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");

    // f32, via the crate's own writer.
    let f32_path = dir.join("rt_f32.npy");
    let values: Vec<f32> = vec![-1.5, 0.0, f32::MIN_POSITIVE, 3.25, 1e30, -7.0];
    write_npy_f32(&f32_path, &[2, 3], &values).expect("write f32");
    let back = read_npy(&f32_path).expect("read f32");
    assert_eq!(back.shape, vec![2, 3]);
    assert_eq!(back.f32_slice().expect("f32 payload"), values.as_slice());
    assert!(back.data_f64.is_none(), "an <f4 file must not populate data_f64");

    // i64 has no writer in the crate, so build the v1.0 file by hand.
    let i64_path = dir.join("rt_i64.npy");
    let ints: Vec<i64> = vec![i64::MIN, -1, 0, 1, i64::MAX];
    let header = format!(
        "{{'descr': '<i8', 'fortran_order': False, 'shape': ({},), }}\n",
        ints.len()
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&u16::try_from(header.len()).expect("header fits").to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in &ints {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&i64_path, &bytes).expect("write i64");
    let back = read_npy(&i64_path).expect("read i64");
    assert_eq!(back.shape, vec![ints.len()]);
    assert_eq!(back.i64_slice().expect("i64 payload"), ints.as_slice());

    std::fs::remove_dir_all(&dir).expect("clean temp dir");
}

/// An unsupported dtype is still refused by name rather than best-effort parsed.
/// Adding `<f8` must not have turned the fallthrough into something lenient.
#[test]
fn an_unsupported_dtype_is_still_refused_and_named() {
    let dir = std::env::temp_dir().join(format!("tessl-npy-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("bad.npy");

    let header = "{'descr': '<c16', 'fortran_order': False, 'shape': (1,), }\n";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&u16::try_from(header.len()).expect("header fits").to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0u8; 16]);
    std::fs::write(&path, &bytes).expect("write");

    let err = read_npy(&path).expect_err("complex128 must be refused");
    assert!(err.contains("<c16"), "refusal must name the dtype, got: {err}");

    std::fs::remove_dir_all(&dir).expect("clean temp dir");
}

/// The Rust f64 reference reproduces the Python-generated golden.
///
/// This is what lets `tests/gdn.rs` judge a kernel at a shape nobody generated a
/// fixture for: the operator is pinned in this crate, not only in another repo's
/// Python. Both sides accumulate in f64 from bit-identical f32 operands, so the
/// only permitted difference is summation order.
#[test]
fn the_rust_f64_reference_reproduces_the_published_golden() {
    let mut worst = (String::new(), 0.0f64);
    for name in case_names() {
        let c = load_case(&name);
        let got = c.run(Rule::Published);
        let (err, mag) = worst_abs(&got, &c.golden);
        let rel = err / mag;
        assert!(
            rel <= REF_REL_BOUND,
            "{}: Rust f64 reference differs from the published golden by {err:.3e} \
             ({rel:.3e} relative to a magnitude of {mag:.4}), over {REF_REL_BOUND:.1e}",
            c.name
        );
        if rel > worst.1 {
            worst = (c.name, rel);
        }
    }
    if worst.0.is_empty() {
        println!(
            "reference vs golden: bit-exact on all {} cases (budget {:.1e})",
            case_names().len(),
            REF_REL_BOUND
        );
    } else {
        println!(
            "worst reference-vs-golden disagreement: {:.3e} on {} (budget {:.1e})",
            worst.1, worst.0, REF_REL_BOUND
        );
    }
}

/// Repo rule 9, with teeth: the two rules are genuinely different operators.
///
/// The rule-9 filename check catches a *mislabelled* fixture. This catches the
/// case that would survive it — an implementation that quietly computes the
/// wrong recurrence. If this test ever passes trivially because the two rules
/// agree, the distinction rule 9 protects has evaporated and the naming
/// convention is guarding nothing.
#[test]
fn the_repo_rule_is_a_different_operator_and_does_not_reproduce_the_golden() {
    let mut smallest_gap = f64::INFINITY;
    let mut single_step_seen = false;

    for name in case_names() {
        let c = load_case(&name);
        let (err, mag) = worst_abs(&c.run(Rule::Repo), &c.golden);
        let rel = err / mag;

        if c.dims.l == 1 {
            // Not a weaker case -- a stronger one. At t=0 the state is zero, so
            // `a * S^T k` and `S^T k` are both zero and the rules are provably
            // identical. Asserting divergence here would be asserting something
            // false; asserting equality pins the mechanism.
            single_step_seen = true;
            assert_eq!(
                err, 0.0,
                "{}: with L=1 the state is zero at the only step, so the two rules \
                 must agree exactly, but they differ by {err:.3e}",
                c.name
            );
            continue;
        }

        assert!(
            rel > 1e-3,
            "{}: the repo rule came within {rel:.3e} of the published golden. \
             Either the rules have stopped differing or the wrong one is being \
             computed -- rule 9 exists because these are different operators.",
            c.name
        );
        smallest_gap = smallest_gap.min(rel);
    }

    assert!(
        single_step_seen,
        "the L=1 case has gone missing; the rules-coincide-at-zero-state branch is \
         no longer exercised and this test has quietly narrowed"
    );
    println!("smallest published-vs-repo divergence (L>1): {smallest_gap:.3e}");
}

/// The clamp the reference applies is the clamp every manifest declares.
///
/// [`GateClamp::PUBLISHED`] is a constant, so without this it could silently go
/// stale against a regenerated corpus. That matters more than it looks: on
/// `L65_tinyalpha` the stored `alpha.npy` is `1e-8`, below the declared `1e-4`
/// floor, and the golden was generated with the clamp applied. Get the clamp
/// wrong and that one case reproduces to 8.4e-5 instead of 1.9e-16 -- close
/// enough to look like rounding, far too large to be it.
#[test]
fn every_manifest_declares_the_clamps_the_reference_applies() {
    let dir = fixture_dir();
    let metas: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("fixture directory must exist")
        .map(|e| e.expect("readable dir entry").path())
        .filter(|p| p.to_string_lossy().ends_with("_meta.json"))
        .collect();
    assert!(!metas.is_empty(), "no per-case manifests found in {}", dir.display());

    // Formatted by the generator with this exact indentation; a reformat should
    // fail loudly here rather than let the constant drift unnoticed.
    let alpha_block = "\"alpha_clamp\": [\n    0.0001,\n    1.0\n  ]";
    let beta_block = "\"beta_clamp\": [\n    0.0,\n    1.0\n  ]";
    assert_eq!(
        GateClamp::PUBLISHED,
        GateClamp {
            alpha: (1e-4, 1.0),
            beta: (0.0, 1.0)
        },
        "the constant changed; update the manifest blocks this test scans for"
    );

    for path in &metas {
        let text = std::fs::read_to_string(path).expect("readable manifest");
        let name = path.file_name().expect("named file").to_string_lossy();
        assert!(
            text.contains(alpha_block),
            "{name} does not declare alpha_clamp [0.0001, 1.0], which is what \
             GateClamp::PUBLISHED applies"
        );
        assert!(
            text.contains(beta_block),
            "{name} does not declare beta_clamp [0.0, 1.0], which is what \
             GateClamp::PUBLISHED applies"
        );
    }
}

/// The clamp is load-bearing, not decorative.
///
/// Runs the one case whose stored gate sits outside the declared floor with the
/// clamp disabled, and requires the result to be *wrong*. If this ever passes
/// with an unclamped gate, the corpus no longer exercises the clamp and
/// `every_manifest_declares_the_clamps_the_reference_applies` is guarding a
/// constant nothing depends on.
#[test]
fn the_alpha_clamp_changes_the_answer_on_the_case_built_to_probe_it() {
    let c = load_case("L65_tinyalpha");
    assert!(
        c.alpha.iter().all(|a| f64::from(*a) < GateClamp::PUBLISHED.alpha.0),
        "L65_tinyalpha is supposed to store an alpha below the clamp floor"
    );

    let unclamped = GateClamp {
        alpha: (0.0, 1.0),
        beta: GateClamp::PUBLISHED.beta,
    };
    let (err, mag) = worst_abs(&sequential_f64(&c.problem(), Rule::Published, unclamped), &c.golden);
    assert!(
        err / mag > 1e-6,
        "skipping the alpha clamp changed nothing ({:.3e} relative); the corpus no \
         longer probes it",
        err / mag
    );
}
