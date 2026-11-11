//! The accounting hook of the buffer pool.
//!
//! The pool ([`crate::GpuRuntime::alloc_buffer_kind`]) allocates by bucket
//! ([`crate::GpuRuntime::allocated_bytes_for`]). A host that keeps one byte
//! ledger for the whole process (ojas's `Budget`) installs a [`PoolLedger`]
//! with [`crate::GpuRuntime::set_pool_ledger`], and the pool then charges what
//! is *resident*, in two balances:
//!
//! - [`LedgerClass::Live`]: buffers handed out, plus buffers whose owner
//!   dropped them and that wait for the GPU to finish (a buffer is charged
//!   from the allocation until it is destroyed or parked on the free list).
//! - [`LedgerClass::Cached`]: buffers parked on the free list.
//!
//! Every charge is the bucket length (`buffer.length()`), taken before
//! `newBufferWithLength`, so what the ledger holds equals what Metal was
//! asked for. A free-list hit moves its bytes from `Cached` to `Live`; no
//! new room is needed for it.
//!
//! # The ladder
//!
//! When a fresh buffer's charge is refused the pool runs a fixed ladder:
//! trim the whole free list (releasing its `Cached` charges), retry the charge
//! once, then refuse with [`LedgerRefused`]. The ladder never waits on the
//! GPU. A caller that wants dropped buffers whose GPU work has finished
//! counted out first calls [`crate::GpuRuntime::reclaim`] before it
//! allocates.
//!
//! # Default
//!
//! No ledger is installed by default; the pool then behaves as it did before
//! this module existed (the only addition is the byte counters behind
//! [`crate::GpuRuntime::pool_resident`]).

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use ojas_core::{Budget, BudgetTrimHandler, OjasError, Reservation};

/// Which pool balance a charge is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LedgerClass {
    /// Handed out, or dropped and waiting for the GPU.
    Live,
    /// Parked on the free list.
    Cached,
}

/// A refused charge: `requested` bytes on top of `live` would pass `cap`.
///
/// The cap is the ledger's; a refusal never shrinks the request. The text is
/// the pool's error string for it, and [`Self::parse`] is its one reader, so a
/// caller deciding "over budget" never matches message fragments of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LedgerRefused {
    pub requested: u64,
    pub cap: u64,
    pub live: u64,
}

impl fmt::Display for LedgerRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "pool ledger refused {} bytes: cap {}, resident {}",
            self.requested, self.cap, self.live
        )
    }
}

impl std::error::Error for LedgerRefused {}

impl LedgerRefused {
    /// Read a pool error string: `Some` only for exactly the text
    /// [`fmt::Display`] builds (the whole string, three decimal numbers).
    pub fn parse(error: &str) -> Option<Self> {
        let rest = error.strip_prefix("pool ledger refused ")?;
        let (requested, rest) = rest.split_once(" bytes: cap ")?;
        let (cap, live) = rest.split_once(", resident ")?;
        // Digits only: `u64::from_str` would also accept a leading `+`. An
        // empty string fails the parse.
        let number = |s: &str| {
            if s.bytes().all(|b| b.is_ascii_digit()) {
                s.parse::<u64>().ok()
            } else {
                None
            }
        };
        Some(Self {
            requested: number(requested)?,
            cap: number(cap)?,
            live: number(live)?,
        })
    }
}

/// A byte ledger the pool charges. Implementations are shared by every pool
/// (every runtime) of the process, so they are `Send + Sync`.
///
/// `release` and the `from` side of `transfer` are always for bytes the pool
/// charged earlier with the same class and length; an implementation may
/// match them exactly.
///
/// Unwinding: the pool assumes a call that panics has changed nothing. A
/// `release` either gives the charge back and returns, or unwinds with the
/// charge still held; `try_charge` unwinds with nothing charged. The pool
/// relies on this to retry: a buffer whose release unwound stays on the free
/// list (or its queue) and is released again by the next trim, drain or Drop.
/// A `release` that panics after it has already decremented its balance is
/// outside this contract and would be released twice by that retry (a
/// ledger that matches releases exactly counts it as unmatched). The pool
/// makes a Live to Cached move as a `release` and a `try_charge`, never one
/// `transfer`, so it always knows which half ran when one unwinds.
///
/// A runtime that is dropped hands back every charge it still holds, and a
/// ledger that panics in one of those releases does not stop the rest: the
/// pool catches each panic, releases everything else, tries the failed
/// releases once more, and then resumes the first panic, so it still reaches
/// whoever dropped the runtime. When the runtime is dropped while the thread
/// is already unwinding, resuming would abort the process, so the pool drops
/// the panic instead and does not retry: the charges whose release panicked
/// stay held.
pub trait PoolLedger: Send + Sync {
    /// Charge `bytes` to `class`, or refuse. Charges nothing on refusal.
    fn try_charge(&self, class: LedgerClass, bytes: u64) -> Result<(), LedgerRefused>;

    /// Give back `bytes` charged to `class`.
    fn release(&self, class: LedgerClass, bytes: u64);

    /// Move `bytes` from `from` to `to`. The default releases, then charges,
    /// so the move needs no room beyond what another holder takes in between.
    ///
    /// On `Err` the `from` charge has been released and nothing is charged to
    /// `to`; the pool destroys the buffer.
    fn transfer(&self, bytes: u64, from: LedgerClass, to: LedgerClass) -> Result<(), LedgerRefused> {
        self.release(from, bytes);
        self.try_charge(to, bytes)
    }
}

/// What one pool holds, as the pool counts it (with or without a ledger).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolResident {
    /// Bucket bytes handed out or waiting for the GPU ([`LedgerClass::Live`]).
    pub live_bytes: u64,
    /// Bucket bytes parked on the free list ([`LedgerClass::Cached`]).
    pub cached_bytes: u64,
}

impl PoolResident {
    /// Everything the pool keeps resident.
    pub fn total(&self) -> u64 {
        self.live_bytes.saturating_add(self.cached_bytes)
    }
}

/// The reservations held for each balance, by the size they were taken at.
///
/// A [`Budget`] is charged through a [`Reservation`] that releases when
/// dropped, and a reservation cannot be split. The pool releases exactly the
/// bucket lengths it charged, so a release finds a reservation of its own size.
#[derive(Default)]
struct Held {
    live: HashMap<u64, Vec<Reservation>>,
    cached: HashMap<u64, Vec<Reservation>>,
    /// Releases for a size with no reservation held: a charge the pool never
    /// made. A correct pool leaves this at zero.
    unmatched: u64,
}

impl Held {
    fn of(&mut self, class: LedgerClass) -> &mut HashMap<u64, Vec<Reservation>> {
        match class {
            LedgerClass::Live => &mut self.live,
            LedgerClass::Cached => &mut self.cached,
        }
    }
}

#[derive(Debug)]
struct NoOpTrim;

impl BudgetTrimHandler for NoOpTrim {
    fn trim_memory(&self, _needed_bytes: u64) -> u64 {
        0
    }
}

/// A [`PoolLedger`] over two children of one [`ojas_core::Budget`]:
///
/// - `live`, the buffers handed out or dropped and waiting for the GPU, and
/// - `cache`, the buffers parked on the runtime's free list,
///
/// so the free-list cache is visible beside the live bytes and both count
/// against the parent's cap.
pub struct OjasBudgetLedger {
    live: Budget,
    cache: Budget,
    held: Mutex<Held>,
}

impl OjasBudgetLedger {
    /// A ledger charging `root`. The live child may take all of `root`'s cap;
    /// the cache child is capped at `cache_cap_bytes`.
    pub fn new(root: &Budget, cache_cap_bytes: u64) -> Arc<Self> {
        let live = root.child(root.cap_bytes());
        let cache = root.child(cache_cap_bytes);

        // Child budgets must not delegate to root's trim handler.
        // BufferPool::charge_fresh already drives its own ladder,
        // so try_charge must fail fast without recursing back to the device thread.
        let noop: Arc<dyn BudgetTrimHandler> = Arc::new(NoOpTrim);
        live.set_trim_handler(Arc::clone(&noop));
        cache.set_trim_handler(noop);

        Arc::new(Self {
            live,
            cache,
            held: Mutex::new(Held::default()),
        })
    }

    /// This ledger as the trait object [`crate::GpuRuntime::set_pool_ledger`] installs.
    pub fn as_pool_ledger(self: &Arc<Self>) -> Arc<dyn PoolLedger> {
        self.clone()
    }

    /// Bytes charged for buffers handed out or waiting for the GPU.
    pub fn live_bytes(&self) -> u64 {
        self.live.live_bytes().unwrap_or(0)
    }

    /// Bytes charged for buffers parked on the runtime's free list.
    pub fn cached_bytes(&self) -> u64 {
        self.cache.live_bytes().unwrap_or(0)
    }

    /// Releases that matched no charge. Zero unless the pool is wrong.
    pub fn unmatched_releases(&self) -> u64 {
        self.lock().unmatched
    }

    /// The error for a pool allocation failure that `error` (a tessl pool
    /// message) reports as a refused charge; `None` for any other message.
    pub fn capacity_error(error: &str) -> Option<OjasError> {
        LedgerRefused::parse(error).map(Self::refusal_error)
    }

    /// A refused charge as the [`OjasError`] a caller sees.
    pub fn refusal_error(refused: LedgerRefused) -> OjasError {
        OjasError::CapacityExceeded {
            requested: refused.requested,
            cap: refused.cap,
            live: refused.live,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Held> {
        self.held.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn budget(&self, class: LedgerClass) -> &Budget {
        match class {
            LedgerClass::Live => &self.live,
            LedgerClass::Cached => &self.cache,
        }
    }

    /// The refusal for a reservation `OjasError` that is not a plain cap
    /// hit (an overflowing counter): numbers that cannot be exceeded.
    fn refused(&self, class: LedgerClass, bytes: u64, error: OjasError) -> LedgerRefused {
        match error {
            OjasError::CapacityExceeded { requested, cap, live } => LedgerRefused { requested, cap, live },
            _ => LedgerRefused {
                requested: bytes,
                cap: self.budget(class).cap_bytes(),
                live: u64::MAX,
            },
        }
    }
}

impl PoolLedger for OjasBudgetLedger {
    fn try_charge(&self, class: LedgerClass, bytes: u64) -> Result<(), LedgerRefused> {
        let reservation = self
            .budget(class)
            .try_reserve(bytes)
            .map_err(|e| self.refused(class, bytes, e))?;
        self.lock().of(class).entry(bytes).or_default().push(reservation);
        Ok(())
    }

    fn release(&self, class: LedgerClass, bytes: u64) {
        // Dropped outside the lock: dropping releases the charge up the chain.
        let reservation = {
            let mut held = self.lock();
            let popped = held.of(class).get_mut(&bytes).and_then(Vec::pop);
            if popped.is_none() {
                held.unmatched += 1;
            }
            popped
        };
        drop(reservation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KIB: u64 = 1024;

    #[test]
    fn a_refusal_reads_back_from_its_text() {
        for (requested, cap, live) in [(0, 0, 0), (16384, 1 << 31, 5), (u64::MAX, u64::MAX, u64::MAX)] {
            let r = LedgerRefused { requested, cap, live };
            assert_eq!(LedgerRefused::parse(&r.to_string()), Some(r), "{r}");
        }
    }

    #[test]
    fn only_the_exact_text_is_a_refusal() {
        for e in [
            "",
            "pool ledger refused",
            "pool ledger refused 1 bytes: cap 2, resident",
            "pool ledger refused 1 bytes: cap 2, resident 3 ",
            " pool ledger refused 1 bytes: cap 2, resident 3",
            "pool ledger refused -1 bytes: cap 2, resident 3",
            "pool ledger refused 1 bytes: cap 2, resident 3x",
            "pool ledger refused 99999999999999999999999 bytes: cap 2, resident 3",
            "runtime is poisoned (pool ledger refused 1 bytes: cap 2, resident 3)",
            "newBuffer(4096) failed",
        ] {
            assert_eq!(LedgerRefused::parse(e), None, "{e:?}");
        }
    }

    #[test]
    fn resident_total_saturates() {
        let r = PoolResident {
            live_bytes: u64::MAX,
            cached_bytes: 1,
        };
        assert_eq!(r.total(), u64::MAX);
    }

    #[test]
    fn charges_and_releases_balance_per_class() {
        let root = Budget::new(64 * KIB);
        let ledger = OjasBudgetLedger::new(&root, 64 * KIB);
        ledger.try_charge(LedgerClass::Live, 16 * KIB).unwrap();
        ledger.try_charge(LedgerClass::Live, 16 * KIB).unwrap();
        ledger.try_charge(LedgerClass::Cached, 8 * KIB).unwrap();
        assert_eq!(ledger.live_bytes(), 32 * KIB);
        assert_eq!(ledger.cached_bytes(), 8 * KIB);
        assert_eq!(root.live_bytes().unwrap(), 40 * KIB, "the parent sees both balances");

        ledger.release(LedgerClass::Live, 16 * KIB);
        ledger.release(LedgerClass::Cached, 8 * KIB);
        assert_eq!(ledger.live_bytes(), 16 * KIB);
        assert_eq!(ledger.cached_bytes(), 0);
        assert_eq!(root.live_bytes().unwrap(), 16 * KIB);
        assert_eq!(ledger.unmatched_releases(), 0);
    }

    #[test]
    fn refusal_is_exactly_at_the_parent_cap_and_charges_nothing() {
        let root = Budget::new(48 * KIB);
        let ledger = OjasBudgetLedger::new(&root, 48 * KIB);
        ledger.try_charge(LedgerClass::Live, 32 * KIB).unwrap();
        ledger.try_charge(LedgerClass::Cached, 16 * KIB).unwrap();
        let refused = ledger.try_charge(LedgerClass::Live, 1).unwrap_err();
        assert_eq!(
            refused,
            LedgerRefused {
                requested: 1,
                cap: 48 * KIB,
                live: 48 * KIB
            }
        );
        assert_eq!(root.live_bytes().unwrap(), 48 * KIB);
        assert_eq!(root.peak_bytes(), 48 * KIB, "a refusal raises no peak");

        ledger.release(LedgerClass::Cached, 16 * KIB);
        assert!(ledger.try_charge(LedgerClass::Live, 16 * KIB + 1).is_err());
        ledger.try_charge(LedgerClass::Live, 16 * KIB).unwrap();
        assert_eq!(root.live_bytes().unwrap(), 48 * KIB);
    }

    #[test]
    fn the_cache_child_has_its_own_cap() {
        let root = Budget::new(64 * KIB);
        let ledger = OjasBudgetLedger::new(&root, 16 * KIB);
        ledger.try_charge(LedgerClass::Cached, 16 * KIB).unwrap();
        let refused = ledger.try_charge(LedgerClass::Cached, 1).unwrap_err();
        assert_eq!(refused.cap, 16 * KIB);
        assert_eq!(refused.live, 16 * KIB);
        ledger.try_charge(LedgerClass::Live, 48 * KIB).unwrap();
    }

    #[test]
    fn transfer_moves_a_charge_without_needing_room() {
        let root = Budget::new(16 * KIB);
        let ledger = OjasBudgetLedger::new(&root, 16 * KIB);
        ledger.try_charge(LedgerClass::Cached, 16 * KIB).unwrap();
        ledger
            .transfer(16 * KIB, LedgerClass::Cached, LedgerClass::Live)
            .unwrap();
        assert_eq!((ledger.live_bytes(), ledger.cached_bytes()), (16 * KIB, 0));
        ledger
            .transfer(16 * KIB, LedgerClass::Live, LedgerClass::Cached)
            .unwrap();
        assert_eq!((ledger.live_bytes(), ledger.cached_bytes()), (0, 16 * KIB));
        assert_eq!(ledger.unmatched_releases(), 0);
    }

    #[test]
    fn a_failed_transfer_has_released_its_source() {
        let root = Budget::new(16 * KIB);
        let ledger = OjasBudgetLedger::new(&root, 8 * KIB);
        ledger.try_charge(LedgerClass::Live, 16 * KIB).unwrap();
        assert!(ledger
            .transfer(16 * KIB, LedgerClass::Live, LedgerClass::Cached)
            .is_err());
        assert_eq!((ledger.live_bytes(), ledger.cached_bytes()), (0, 0));
        assert_eq!(root.live_bytes().unwrap(), 0);
    }

    #[test]
    fn a_release_without_a_charge_is_counted_not_applied() {
        let root = Budget::new(16 * KIB);
        let ledger = OjasBudgetLedger::new(&root, 16 * KIB);
        ledger.try_charge(LedgerClass::Live, 4 * KIB).unwrap();
        ledger.release(LedgerClass::Live, 8 * KIB);
        ledger.release(LedgerClass::Cached, 4 * KIB);
        assert_eq!(ledger.unmatched_releases(), 2);
        assert_eq!(ledger.live_bytes(), 4 * KIB, "an unmatched release frees nothing");
    }

    fn numbers(error: Option<OjasError>) -> Option<(u64, u64, u64)> {
        match error {
            Some(OjasError::CapacityExceeded { requested, cap, live }) => Some((requested, cap, live)),
            _ => None,
        }
    }

    #[test]
    fn a_refusal_is_a_capacity_error_with_its_numbers() {
        let refused = LedgerRefused {
            requested: 256,
            cap: 1 << 20,
            live: 1 << 20,
        };
        let want = Some((256, 1 << 20, 1 << 20));
        assert_eq!(numbers(Some(OjasBudgetLedger::refusal_error(refused))), want);
        assert_eq!(numbers(OjasBudgetLedger::capacity_error(&refused.to_string())), want);
        assert!(OjasBudgetLedger::capacity_error("newBuffer(4096) failed").is_none());
    }

    #[test]
    fn an_overflowing_counter_is_refused_with_numbers_that_cannot_be_exceeded() {
        let root = Budget::new(u64::MAX);
        let ledger = OjasBudgetLedger::new(&root, u64::MAX);
        ledger.try_charge(LedgerClass::Live, u64::MAX).unwrap();
        let refused = ledger.try_charge(LedgerClass::Live, 1).unwrap_err();
        assert_eq!(refused.requested, 1);
        assert_eq!(refused.live, u64::MAX);
    }
}
