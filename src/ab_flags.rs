//! GEMM / encode A/B flags (env, read once, then overridable for inference).
//!
//! Preserves metal-native Audit 4/6 lessons that affect the shared runtime:
//! - always-on Device barrier after each dispatch (golden-safe)
//! - f32 GEMM interior offset tiles off by default
//! - TensorOps multiply_accumulate off by default
//!
//! Env names are canonically `TESSL_*`. The legacy `METAL_RUNTIME_*`,
//! `METAL_NATIVE_*` and (for coarse barriers only) `GEMMA_METAL_*` spellings are
//! still read, after the canonical one, so scripts written against either former
//! crate keep working after the rename. The first name whose value *parses* wins;
//! a name set to anything outside the recognised spellings below is skipped and
//! the next name is tried, and if no name parses the compiled-in default applies.
//! The recognised spellings are exact and case-sensitive (`1/true/TRUE/yes` and
//! `0/false/FALSE/no`) — `Yes`, `True` and `NO` are *not* recognised and fall
//! through to the default.

use std::sync::atomic::{AtomicI8, Ordering};
use std::sync::OnceLock;

// Env name lists, canonical `TESSL_*` first and legacy spellings after. They are
// named constants rather than inline literals because the *order* is the part
// that has silently broken before: `coarse_barriers` used to read
// `GEMMA_METAL_COARSE_BARRIERS` ahead of its own canonical name, so the one flag
// whose doc promised "TESSL_* first" was the one flag that did not do it. A test
// pins the invariant for every list at once.
const ENV_GEMM_INTERIOR: &[&str] = &[
    "TESSL_GEMM_INTERIOR",
    "METAL_RUNTIME_GEMM_INTERIOR",
    "METAL_NATIVE_GEMM_INTERIOR",
];
const ENV_GEMM_ACCUM: &[&str] = &[
    "TESSL_GEMM_ACCUM",
    "METAL_RUNTIME_GEMM_ACCUM",
    "METAL_NATIVE_GEMM_ACCUM",
];
const ENV_GEMM_ACCUM_DX: &[&str] = &[
    "TESSL_GEMM_ACCUM_DX",
    "METAL_RUNTIME_GEMM_ACCUM_DX",
    "METAL_NATIVE_GEMM_ACCUM_DX",
];
const ENV_HAZARD_BARRIERS: &[&str] = &[
    "TESSL_HAZARD_BARRIERS",
    "METAL_RUNTIME_HAZARD_BARRIERS",
    "METAL_NATIVE_HAZARD_BARRIERS",
];
// `GEMMA_METAL_COARSE_BARRIERS` keeps its slot ahead of `METAL_RUNTIME_*` (that
// was its relative order before) but now sits behind the canonical name; nothing
// in this workspace sets it, so the reorder can only matter to a caller that sets
// both names to conflicting values, and for that caller `TESSL_*` winning is what
// the module doc has always promised.
const ENV_COARSE_BARRIERS: &[&str] = &[
    "TESSL_COARSE_BARRIERS",
    "GEMMA_METAL_COARSE_BARRIERS",
    "METAL_RUNTIME_COARSE_BARRIERS",
];

/// The recognised spellings, exact and case-sensitive. `None` means "this value
/// does not decide the flag" — for an unset name *and* for a set-but-unrecognised
/// one such as `Yes` or `2`.
fn parse_truthy(value: Option<&str>) -> Option<bool> {
    match value {
        Some("1") | Some("true") | Some("TRUE") | Some("yes") => Some(true),
        Some("0") | Some("false") | Some("FALSE") | Some("no") => Some(false),
        _ => None,
    }
}

/// Walk `names` in order and take the first value that parses; `None` if none
/// does, which leaves the compiled-in default in place.
///
/// `lookup` is a parameter rather than a hard-wired `std::env::var` so the
/// precedence rule can be tested against a table. Mutating the real environment
/// from a test is not an option here: `setenv` races every concurrent `getenv`,
/// and this crate reads env lazily from inside GPU paths ([`flags`] latches on
/// the first `gemm_accum()` call, which happens mid-GEMM), so a test that set a
/// variable could corrupt an unrelated test's numerics rather than its own.
///
/// Lazy on purpose: `find_map` stops at the first name that parses, so a later
/// name is not even looked up.
fn resolve(names: &[&str], lookup: impl Fn(&str) -> Option<String>) -> Option<bool> {
    names
        .iter()
        .find_map(|name| parse_truthy(lookup(name).as_deref()))
}

pub(crate) fn env_truthy(names: &[&str]) -> Option<bool> {
    resolve(names, |name| std::env::var(name).ok())
}

fn flags() -> &'static (bool, bool, bool) {
    static FLAGS: OnceLock<(bool, bool, bool)> = OnceLock::new();
    FLAGS.get_or_init(|| {
        let gemm_interior = env_truthy(ENV_GEMM_INTERIOR).unwrap_or(false);
        let gemm_accum = env_truthy(ENV_GEMM_ACCUM).unwrap_or(false);
        // dX-only accumulate. Separate from `gemm_accum` because every NT-accum
        // call site targets a fresh pre-zeroed activation-grad buffer, never a
        // weight bank — the case the full flag was turned off for.
        let gemm_accum_dx = env_truthy(ENV_GEMM_ACCUM_DX).unwrap_or(false);
        (gemm_interior, gemm_accum, gemm_accum_dx)
    })
}

/// -1 = unset (read env), 0 = always-on barriers, 1 = skip auto (hazard mode).
static HAZARD_BARRIERS: AtomicI8 = AtomicI8::new(-1);

/// Force skip/enable always-on Device barrier. Call before first encode for decode.
/// **Must** insert explicit [`crate::dispatch::Binder::barrier`] at RAW edges when
/// enabling (skip_auto=true).
pub fn set_hazard_barriers(skip_auto: bool) {
    HAZARD_BARRIERS.store(if skip_auto { 1 } else { 0 }, Ordering::Relaxed);
}

/// True once [`set_hazard_barriers`] has been called (or env was read by
/// [`hazard_barriers`]). Used by GPU init to avoid clobbering an explicit choice.
pub fn hazard_barriers_explicitly_set() -> bool {
    HAZARD_BARRIERS.load(Ordering::Relaxed) >= 0
}

/// Use interior offset tile extents in f32 GEMM. Default: false.
pub fn gemm_interior_offsets() -> bool {
    flags().0
}

/// Skip always-on Dispatch→Dispatch Device barrier after every `Binder::dispatch`.
/// Packed multi-dispatch ops still insert explicit barriers. Default: false
/// (golden-safe). **Do not enable as default** without RAW-edge barriers.
pub fn hazard_barriers() -> bool {
    let v = HAZARD_BARRIERS.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let from_env = env_truthy(ENV_HAZARD_BARRIERS).unwrap_or(false);
    HAZARD_BARRIERS.store(if from_env { 1 } else { 0 }, Ordering::Relaxed);
    from_env
}

/// Use TensorOps `multiply_accumulate` for accumulate GEMMs. Default: false.
pub fn gemm_accum() -> bool {
    flags().1
}

/// Accumulate-mode for the **dX** NT path only. Every caller accumulates into a
/// fresh pre-zeroed activation-grad buffer, never a weight bank, so this is safe
/// to enable independently of [`gemm_accum`].
pub fn gemm_accum_dx() -> bool {
    flags().2
}

/// -1 unset, 0 fine-grained RAW barriers, 1 phase-coarsened (default when hazard).
static COARSE_BARRIERS: AtomicI8 = AtomicI8::new(-1);

/// Default when no `*_COARSE_BARRIERS` name parses: inherit [`hazard_barriers`].
/// Deliberately not a constant — coarsening is only ever a question in hazard
/// mode. With always-on Device barriers there is no per-RAW barrier left to
/// coarsen, so the inherited `false` says "nothing to do" rather than "fine
/// grained"; pinning it to a constant `true` would arm phase-only barriers on a
/// runtime that had not opted into hazard mode at all.
fn coarse_default(from_env: Option<bool>, hazard: bool) -> bool {
    from_env.unwrap_or(hazard)
}

/// Coarsen decode RAW barriers to phase edges (fewer Device drains).
/// Default: on when [`hazard_barriers`] is true (see `coarse_default`, private).
/// Override with `TESSL_COARSE_BARRIERS=0|1` (legacy:
/// `GEMMA_METAL_COARSE_BARRIERS`, `METAL_RUNTIME_COARSE_BARRIERS`).
pub fn coarse_barriers() -> bool {
    let v = COARSE_BARRIERS.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let on = coarse_default(env_truthy(ENV_COARSE_BARRIERS), hazard_barriers());
    COARSE_BARRIERS.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

/// Pure core of [`need_barrier`], split out so the truth table can be pinned by
/// tests without latching or mutating the process-global flags.
///
/// `coarse` is a closure, not a `bool`, because the laziness is load-bearing:
/// [`coarse_barriers`] latches its value on first read and inherits whatever
/// [`hazard_barriers`] said *at that moment*. Evaluating it while hazard mode is
/// still off would freeze `coarse = false` for a caller that turns hazard mode on
/// later (gemma-metal's dflash does exactly that around `set_hazard_barriers`),
/// silently converting phase-coarsened decode into fine-grained decode.
fn need_barrier_from(hazard: bool, coarse: impl FnOnce() -> bool, phase_edge: bool) -> bool {
    if !hazard {
        // Always-on Device barriers are still being emitted after every dispatch,
        // so an explicit RAW barrier here would only add a redundant drain.
        return false;
    }
    if coarse() {
        phase_edge
    } else {
        true
    }
}

/// Explicit RAW barrier needed for a phase edge. In fine mode: always when hazard.
/// In coarse mode: only when `phase_edge` is true (major producer→consumer).
pub fn need_barrier(phase_edge: bool) -> bool {
    need_barrier_from(hazard_barriers(), coarse_barriers, phase_edge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    // Concurrency policy for this module's tests, stated once.
    //
    // These flags are process-global (`OnceLock` + `AtomicI8`) with an env
    // fallback that latches on first read, and cargo runs tests as parallel
    // threads in one process. So:
    //
    // * Nothing here mutates the environment. `setenv` is not thread-safe against
    //   a concurrent `getenv`, and this crate reads env lazily from inside GPU
    //   paths (`flags()` latches on the first `gemm_accum()`, mid-GEMM), so a
    //   test that set a variable could corrupt an unrelated test's numerics.
    //   Parsing and precedence are covered through `parse_truthy` / `resolve`
    //   with a table-driven lookup instead, which is stronger anyway: it can
    //   assert which names were consulted, which env vars cannot.
    // * Nothing here calls `set_hazard_barriers`. Flipping it would turn the
    //   always-on Device barrier off for every *other* test in this binary — the
    //   GEMM suites included — and produce wrong results, not just a flaky flag
    //   test. That setter is left to integration callers that own the process.
    // * Decision logic (`coarse_default`, `need_barrier_from`) is tested pure.
    // * The two tests that touch the live globals only *read* them, and a read is
    //   idempotent: the first latches, every later one returns the same value, so
    //   they are order-independent with respect to every other test here.

    /// A `resolve` lookup over a fixed table, recording which names it was asked
    /// for so laziness can be asserted.
    fn table<'a>(
        entries: &'a [(&'a str, &'a str)],
        seen: &'a Cell<usize>,
    ) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            seen.set(seen.get() + 1);
            entries
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| (*v).to_string())
        }
    }

    /// The exact accepted spellings. `Yes`/`True`/`NO` are asserted to be
    /// *unrecognised*: a caller who writes one silently gets the compiled-in
    /// default with no diagnostic, and that asymmetry is worth failing a test
    /// over if it ever changes without the module doc changing with it.
    #[test]
    fn parse_truthy_accepts_only_the_documented_spellings() {
        let cases: &[(Option<&str>, Option<bool>)] = &[
            (Some("1"), Some(true)),
            (Some("true"), Some(true)),
            (Some("TRUE"), Some(true)),
            (Some("yes"), Some(true)),
            (Some("0"), Some(false)),
            (Some("false"), Some(false)),
            (Some("FALSE"), Some(false)),
            (Some("no"), Some(false)),
            (Some("True"), None),
            (Some("Yes"), None),
            (Some("YES"), None),
            (Some("NO"), None),
            (Some("2"), None),
            (Some("on"), None),
            (Some(""), None),
            (None, None),
        ];
        for (value, expected) in cases {
            assert_eq!(parse_truthy(*value), *expected, "parse_truthy({value:?})");
        }
    }

    /// Precedence is positional, not "most specific": whichever *parsing* name
    /// comes first wins, including when it says `false`. An unparsable value in
    /// the first slot does not veto the rest — it falls through to the next name.
    #[test]
    fn resolve_takes_the_first_name_that_parses() {
        const NAMES: &[&str] = &["FIRST", "SECOND", "THIRD"];
        /// One case: the names that are "set", and what `resolve` must return.
        type Case<'a> = (&'a [(&'a str, &'a str)], Option<bool>);
        let cases: &[Case] = &[
            (&[("FIRST", "0"), ("SECOND", "1")], Some(false)),
            (&[("FIRST", "1"), ("SECOND", "0")], Some(true)),
            (&[("SECOND", "1")], Some(true)),
            (&[("THIRD", "0")], Some(false)),
            (&[("FIRST", "junk"), ("SECOND", "0")], Some(false)),
            (&[("FIRST", "junk"), ("THIRD", "1")], Some(true)),
            (&[("FIRST", "junk")], None),
            (&[("FIRST", "junk"), ("SECOND", "Yes")], None),
            (&[], None),
        ];
        for (entries, expected) in cases {
            let seen = Cell::new(0);
            assert_eq!(
                resolve(NAMES, table(entries, &seen)),
                *expected,
                "resolve over {entries:?}"
            );
        }
    }

    /// A name that already decided the flag must end the walk. This is not
    /// cosmetic: `resolve` is the only reader of these variables, and stopping
    /// early is what keeps a legacy name from being consulted at all once the
    /// canonical one has answered.
    #[test]
    fn resolve_stops_at_the_first_parsing_name() {
        const NAMES: &[&str] = &["FIRST", "SECOND", "THIRD"];
        let seen = Cell::new(0);
        assert_eq!(resolve(NAMES, table(&[("FIRST", "1")], &seen)), Some(true));
        assert_eq!(seen.get(), 1, "later names must not be looked up");

        let seen = Cell::new(0);
        assert_eq!(
            resolve(NAMES, table(&[("SECOND", "0")], &seen)),
            Some(false)
        );
        assert_eq!(seen.get(), 2, "must stop once SECOND answers");

        let seen = Cell::new(0);
        assert_eq!(resolve(NAMES, table(&[], &seen)), None);
        assert_eq!(seen.get(), 3, "with no answer every name must be tried");
    }

    /// The regression this module has actually suffered: `coarse_barriers` read
    /// `GEMMA_METAL_COARSE_BARRIERS` ahead of `TESSL_COARSE_BARRIERS`, so the one
    /// flag whose doc promised "canonical name first" was the one flag that did
    /// not do it. Every list must lead with its `TESSL_*` name.
    #[test]
    fn every_flag_reads_its_canonical_tessl_name_first() {
        let lists: &[(&str, &[&str])] = &[
            ("gemm_interior", ENV_GEMM_INTERIOR),
            ("gemm_accum", ENV_GEMM_ACCUM),
            ("gemm_accum_dx", ENV_GEMM_ACCUM_DX),
            ("hazard_barriers", ENV_HAZARD_BARRIERS),
            ("coarse_barriers", ENV_COARSE_BARRIERS),
        ];
        for (flag, names) in lists {
            assert!(!names.is_empty(), "{flag} has no env names");
            assert!(
                names[0].starts_with("TESSL_"),
                "{flag} reads {} before its canonical TESSL_* name",
                names[0]
            );
            for (i, name) in names.iter().enumerate() {
                assert!(
                    !names[i + 1..].contains(name),
                    "{flag} lists {name} twice; the duplicate can never win"
                );
                assert!(
                    i == 0 || !name.starts_with("TESSL_"),
                    "{flag} has a second canonical name {name}; only slot 0 is canonical"
                );
            }
        }
    }

    /// Pinned because it is a live quirk, not an accident: with no override the
    /// flag inherits `hazard_barriers` rather than a constant. An explicit value
    /// wins in both directions, including turning coarsening *off* while hazard
    /// mode is on.
    #[test]
    fn coarse_default_inherits_hazard_and_yields_to_an_explicit_value() {
        assert!(
            coarse_default(None, true),
            "hazard on must coarsen by default"
        );
        assert!(
            !coarse_default(None, false),
            "with always-on barriers there is nothing to coarsen"
        );
        assert!(!coarse_default(Some(false), true), "explicit 0 must disarm");
        assert!(coarse_default(Some(true), false), "explicit 1 must arm");
    }

    /// The whole point of the module: whether a device barrier is emitted at all.
    /// An inversion here corrupts GEMM output instead of failing loudly, so the
    /// full truth table is pinned.
    #[test]
    fn need_barrier_truth_table() {
        for phase_edge in [false, true] {
            assert!(
                !need_barrier_from(false, || true, phase_edge),
                "hazard off: the always-on barrier already covers this edge"
            );
            assert!(
                !need_barrier_from(false, || false, phase_edge),
                "hazard off: the always-on barrier already covers this edge"
            );
            assert!(
                need_barrier_from(true, || false, phase_edge),
                "hazard + fine grained: every RAW edge needs an explicit barrier"
            );
            assert_eq!(
                need_barrier_from(true, || true, phase_edge),
                phase_edge,
                "hazard + coarse: only major producer->consumer edges"
            );
        }
    }

    /// `coarse_barriers` latches on first read and inherits hazard's value at
    /// that instant, so reading it while hazard is off would freeze `false` for a
    /// caller that enables hazard mode afterwards (gemma-metal's dflash brackets
    /// work with `set_hazard_barriers`). `need_barrier` must short-circuit before
    /// touching it.
    #[test]
    fn need_barrier_does_not_read_coarse_when_hazard_is_off() {
        let read = Cell::new(false);
        let probe = || {
            read.set(true);
            true
        };
        assert!(!need_barrier_from(false, probe, true));
        assert!(
            !read.get(),
            "need_barrier latched COARSE_BARRIERS while hazard mode was off"
        );

        let read_on = Cell::new(false);
        let probe_on = || {
            read_on.set(true);
            true
        };
        assert!(need_barrier_from(true, probe_on, true));
        assert!(read_on.get(), "hazard on must consult the coarse flag");
    }

    /// Live globals, read-only: the env fallback is latched exactly once and
    /// every later read returns the same answer, which is what
    /// `hazard_barriers_explicitly_set` reports to GPU init so it does not
    /// clobber an explicit choice.
    #[test]
    fn hazard_barriers_latches_once_and_reports_itself_set() {
        let expected = env_truthy(ENV_HAZARD_BARRIERS).unwrap_or(false);
        let first = hazard_barriers();
        assert_eq!(
            first, expected,
            "hazard_barriers disagrees with its own env name list"
        );
        assert!(
            hazard_barriers_explicitly_set(),
            "reading the flag must latch it; GPU init keys off this"
        );
        assert_eq!(hazard_barriers(), first, "latched value must be stable");
    }

    /// Same, one level up: the live `coarse_barriers` path must agree with the
    /// pure default it is built from, so a future edit cannot change the wiring
    /// (env list, inherit rule) without failing here as well.
    #[test]
    fn coarse_barriers_live_path_matches_its_pure_default() {
        let hazard = hazard_barriers();
        let expected = coarse_default(env_truthy(ENV_COARSE_BARRIERS), hazard);
        let first = coarse_barriers();
        assert_eq!(
            first, expected,
            "live coarse_barriers diverged from coarse_default"
        );
        assert_eq!(coarse_barriers(), first, "latched value must be stable");
        assert_eq!(
            need_barrier(true),
            hazard,
            "phase edge: hazard mode decides, in both coarse and fine mode"
        );
        assert_eq!(
            need_barrier(false),
            hazard && !first,
            "non-phase edge: only fine-grained hazard mode needs a barrier"
        );
    }
}
