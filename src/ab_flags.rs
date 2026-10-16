//! GEMM / encode A/B flags (env, read once, then overridable for inference).
//!
//! The defaults below are conservative on purpose. Each one was measured to
//! matter, and the unsafe direction is the one that looks like free speed:
//! - always-on Device barrier after each dispatch (golden-safe)
//!
//! Env names are canonically `TESSL_*`. The legacy `METAL_RUNTIME_*` spelling is
//! still read, after the canonical one. The first name whose value *parses* wins;
//! a name set to anything outside the recognised spellings below is skipped and
//! the next name is tried, and if no name parses the compiled-in default applies.
//! The recognised spellings are exact and case-sensitive (`1/true/TRUE/yes` and
//! `0/false/FALSE/no`) — `Yes`, `True` and `NO` are *not* recognised and fall
//! through to the default.

use std::sync::atomic::{AtomicI8, Ordering};

// Env name lists, canonical `TESSL_*` first and the legacy spelling after. They
// are named constants rather than inline literals because the *order* is the
// part that has silently broken before (`coarse_barriers` once read a legacy
// name ahead of its canonical one). A test pins the invariant for every list at
// once.
const ENV_HAZARD_BARRIERS: &[&str] = &["TESSL_HAZARD_BARRIERS", "METAL_RUNTIME_HAZARD_BARRIERS"];
const ENV_COARSE_BARRIERS: &[&str] = &["TESSL_COARSE_BARRIERS", "METAL_RUNTIME_COARSE_BARRIERS"];

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
/// and this crate reads env lazily from inside GPU paths ([`hazard_barriers`]
/// latches on the first dispatch that asks), so a test that set a variable
/// could corrupt an unrelated test's numerics rather than its own.
///
/// Lazy on purpose: `find_map` stops at the first name that parses, so a later
/// name is not even looked up.
fn resolve(names: &[&str], lookup: impl Fn(&str) -> Option<String>) -> Option<bool> {
    names.iter().find_map(|name| parse_truthy(lookup(name).as_deref()))
}

pub(crate) fn env_truthy(names: &[&str]) -> Option<bool> {
    resolve(names, |name| std::env::var(name).ok())
}

/// One process-global tri-state flag: unset (env not read yet), off, on. Every `*_enabled()` getter in
/// the crate is this struct plus its env name list, so the latch and the
/// override rules live once. A `set` always wins over the env.
pub(crate) struct TriFlag(AtomicI8);

impl TriFlag {
    const UNSET: i8 = -1;

    pub(crate) const fn new() -> Self {
        Self(AtomicI8::new(Self::UNSET))
    }

    /// Force on/off, overriding the env.
    pub(crate) fn set(&self, on: bool) {
        self.0.store(i8::from(on), Ordering::Relaxed);
    }

    /// True once forced, or once the env was read.
    pub(crate) fn is_latched(&self) -> bool {
        self.0.load(Ordering::Relaxed) >= 0
    }

    /// Raw state, for a test guard to save and [`Self::restore`].
    #[cfg(test)]
    pub(crate) fn raw(&self) -> i8 {
        self.0.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn restore(&self, raw: i8) {
        self.0.store(raw, Ordering::Relaxed);
    }

    /// The forced value, else the first parsing name in `names`, else
    /// `default()`; the answer is latched on the first read, so `default` runs
    /// at most once.
    pub(crate) fn get(&self, names: &[&str], default: impl FnOnce() -> bool) -> bool {
        let v = self.0.load(Ordering::Relaxed);
        if v >= 0 {
            return v == 1;
        }
        let on = env_truthy(names).unwrap_or_else(default);
        self.set(on);
        on
    }
}

/// Unset (read env), off = always-on barriers, on = skip auto (hazard mode).
static HAZARD_BARRIERS: TriFlag = TriFlag::new();

/// Force skip/enable always-on Device barrier. Call before first encode for decode.
/// **Must** insert explicit [`crate::dispatch::Binder::barrier`] at RAW edges when
/// enabling (skip_auto=true).
pub fn set_hazard_barriers(skip_auto: bool) {
    HAZARD_BARRIERS.set(skip_auto);
}

/// True once [`set_hazard_barriers`] has been called (or env was read by
/// [`hazard_barriers`]). Used by GPU init to avoid clobbering an explicit choice.
pub fn hazard_barriers_explicitly_set() -> bool {
    HAZARD_BARRIERS.is_latched()
}

/// Skip always-on Dispatch→Dispatch Device barrier after every `Binder::dispatch`.
/// Packed multi-dispatch ops still insert explicit barriers. Default: false
/// (golden-safe). **Do not enable as default** without RAW-edge barriers.
pub fn hazard_barriers() -> bool {
    HAZARD_BARRIERS.get(ENV_HAZARD_BARRIERS, || false)
}

/// Unset, off = fine-grained RAW barriers, on = phase-coarsened (default when hazard).
static COARSE_BARRIERS: TriFlag = TriFlag::new();

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
/// `METAL_RUNTIME_COARSE_BARRIERS`).
pub fn coarse_barriers() -> bool {
    // `get` is latching and its default runs once: coarse inherits the hazard
    // flag as of this first read (see `need_barrier_from`).
    COARSE_BARRIERS.get(ENV_COARSE_BARRIERS, || coarse_default(None, hazard_barriers()))
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
    // These flags are process-global (`AtomicI8`) with an env
    // fallback that latches on first read, and cargo runs tests as parallel
    // threads in one process. So:
    //
    // * Nothing here mutates the environment. `setenv` is not thread-safe against
    //   a concurrent `getenv`, and this crate reads env lazily from inside GPU
    //   paths (`hazard_barriers()` latches on the first dispatch that asks),
    //   so a test that set a variable could corrupt an unrelated test's numerics.
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

    /// `TriFlag` on private instances, with names no env sets, so nothing here
    /// touches the real flags or the environment.
    #[test]
    fn tri_flag_latches_and_forces() {
        const NONE: &[&str] = &["TESSL_TRI_FLAG_TEST_UNSET_A", "TESSL_TRI_FLAG_TEST_UNSET_B"];
        let f = TriFlag::new();
        assert!(!f.is_latched());
        assert!(f.get(NONE, || true), "no name parses: the default decides");
        assert!(f.is_latched());
        assert!(f.get(NONE, || false), "and the first answer is latched");
        f.set(false);
        assert!(!f.get(NONE, || true), "a forced value beats the default");
    }

    /// A `resolve` lookup over a fixed table, recording which names it was asked
    /// for so laziness can be asserted.
    fn table<'a>(entries: &'a [(&'a str, &'a str)], seen: &'a Cell<usize>) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            seen.set(seen.get() + 1);
            entries.iter().find(|(k, _)| *k == name).map(|(_, v)| (*v).to_string())
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
        assert_eq!(resolve(NAMES, table(&[("SECOND", "0")], &seen)), Some(false));
        assert_eq!(seen.get(), 2, "must stop once SECOND answers");

        let seen = Cell::new(0);
        assert_eq!(resolve(NAMES, table(&[], &seen)), None);
        assert_eq!(seen.get(), 3, "with no answer every name must be tried");
    }

    /// The regression this module has actually suffered: `coarse_barriers` read
    /// a legacy name ahead of `TESSL_COARSE_BARRIERS`, so the one flag whose doc
    /// promised "canonical name first" was the one flag that did not do it.
    /// Every list must lead with its `TESSL_*` name.
    #[test]
    fn every_flag_reads_its_canonical_tessl_name_first() {
        let lists: &[(&str, &[&str])] = &[
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
        assert!(coarse_default(None, true), "hazard on must coarsen by default");
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
        assert_eq!(first, expected, "hazard_barriers disagrees with its own env name list");
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
        assert_eq!(first, expected, "live coarse_barriers diverged from coarse_default");
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
