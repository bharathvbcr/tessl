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

use std::fmt;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
