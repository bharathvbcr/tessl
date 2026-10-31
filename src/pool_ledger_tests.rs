//! The pool's ledger hook against a ledger that records every call.
//!
//! A child of `runtime` so it can reach the private pool and the lifetime
//! event: the tests make a drop "finished" by hand where a real command buffer
//! could not be held still. Each test makes its own `GpuRuntime` and needs a
//! Metal device.

use super::*;
use crate::pool_ledger::LedgerRefused;

#[derive(Default)]
struct Books {
    live: u64,
    cached: u64,
    /// Releases for more than the balance held: a charge the pool never made
    /// or made twice.
    unmatched: u64,
    charges: u64,
}

/// A ledger with one cap over `live + cached`, like a `Budget` with two
/// children of the same parent.
struct Fake {
    cap: u64,
    books: Mutex<Books>,
}

impl Fake {
    fn new(cap: u64) -> Arc<Self> {
        Arc::new(Self {
            cap,
            books: Mutex::new(Books::default()),
        })
    }
    fn live(&self) -> u64 {
        self.books.lock().unwrap().live
    }
    fn cached(&self) -> u64 {
        self.books.lock().unwrap().cached
    }
    fn unmatched(&self) -> u64 {
        self.books.lock().unwrap().unmatched
    }
    fn charges(&self) -> u64 {
        self.books.lock().unwrap().charges
    }
}

impl PoolLedger for Fake {
    fn try_charge(&self, class: LedgerClass, bytes: u64) -> Result<(), LedgerRefused> {
        let mut b = self.books.lock().unwrap();
        let resident = b.live + b.cached;
        if resident.checked_add(bytes).is_none_or(|n| n > self.cap) {
            return Err(LedgerRefused {
                requested: bytes,
                cap: self.cap,
                live: resident,
            });
        }
        match class {
            LedgerClass::Live => b.live += bytes,
            LedgerClass::Cached => b.cached += bytes,
        }
        b.charges += 1;
        Ok(())
    }

    fn release(&self, class: LedgerClass, bytes: u64) {
        let mut b = self.books.lock().unwrap();
        let slot = match class {
            LedgerClass::Live => &mut b.live,
            LedgerClass::Cached => &mut b.cached,
        };
        if *slot < bytes {
            *slot = 0;
            b.unmatched += 1;
        } else {
            *slot -= bytes;
        }
    }
}

fn bucket(n: usize, kind: BufferKind) -> u64 {
    GpuRuntime::allocated_bytes_for(n, kind)
}

fn addr(b: &crate::tensor::GpuBuffer) -> usize {
    b.metal().contents().as_ptr() as usize
}

/// A runtime with `ledger` installed.
fn runtime_with(ledger: &Arc<Fake>) -> Arc<GpuRuntime> {
    let rt = GpuRuntime::new().unwrap();
    let as_dyn: Arc<dyn PoolLedger> = ledger.clone();
    rt.set_pool_ledger(Some(as_dyn)).unwrap();
    rt
}

/// What a drop followed by an allocation does when the GPU is idle: the
/// dropped buffers leave the live balance.
fn settle(rt: &GpuRuntime) {
    rt.recycle_if_idle().unwrap();
}

const SIZES: [(usize, BufferKind); 7] = [
    (1, BufferKind::Cold),
    (5_000, BufferKind::Cold),
    (300_000, BufferKind::Cold),
    (1 << 20, BufferKind::Cold),
    ((1 << 20) + 1, BufferKind::Cold),
    (100_000, BufferKind::Hot),
    (3 << 20, BufferKind::Hot),
];

#[test]
fn every_charge_is_the_allocated_bucket() {
    let ledger = Fake::new(u64::MAX);
    let rt = runtime_with(&ledger);
    let mut held = Vec::new();
    let mut total = 0u64;
    for (n, kind) in SIZES {
        held.push(rt.alloc_buffer_kind(n, kind).unwrap());
        total += bucket(n, kind);
        assert_eq!(ledger.live(), total, "{n} {kind:?}");
        assert_eq!(rt.pool_resident().live_bytes, total, "{n} {kind:?}");
        assert_eq!(ledger.cached(), 0);
    }
    // The charge is the length Metal was asked for, not the request.
    let lengths: u64 = held.iter().map(|b| b.metal().length() as u64).sum();
    assert_eq!(lengths, total);

    drop(held);
    settle(&rt);
    let cold: u64 = SIZES
        .iter()
        .filter(|(_, k)| *k == BufferKind::Cold)
        .map(|&(n, k)| bucket(n, k))
        .sum();
    // Cold buffers park on the free list (still resident, now cached); Hot
    // ones are destroyed.
    assert_eq!(ledger.live(), 0);
    assert_eq!(ledger.cached(), cold);
    assert_eq!(
        rt.pool_resident(),
        PoolResident {
            live_bytes: 0,
            cached_bytes: cold
        }
    );
    assert_eq!(ledger.unmatched(), 0);
}

#[test]
fn refusal_happens_exactly_at_the_bucket_total() {
    let (a, b) = (1 << 20, 100_000);
    let total = bucket(a, BufferKind::Cold) + bucket(b, BufferKind::Hot);

    // One byte short of the total: the second buffer is refused.
    let short = Fake::new(total - 1);
    let rt = runtime_with(&short);
    let _a = rt.alloc_buffer_kind(a, BufferKind::Cold).unwrap();
    let err = rt.alloc_buffer_kind(b, BufferKind::Hot).map(|_| ()).unwrap_err();
    assert_eq!(
        LedgerRefused::parse(&err),
        Some(LedgerRefused {
            requested: bucket(b, BufferKind::Hot),
            cap: total - 1,
            live: bucket(a, BufferKind::Cold)
        }),
        "{err}"
    );
    assert_eq!(short.live(), bucket(a, BufferKind::Cold), "a refusal charges nothing");

    // Exactly the total: both fit, and the next smallest bucket does not.
    let exact = Fake::new(total);
    let rt = runtime_with(&exact);
    let _a = rt.alloc_buffer_kind(a, BufferKind::Cold).unwrap();
    let _b = rt.alloc_buffer_kind(b, BufferKind::Hot).unwrap();
    assert_eq!(exact.live(), total);
    let err = rt.alloc_buffer_kind(1, BufferKind::Cold).map(|_| ()).unwrap_err();
    assert_eq!(
        LedgerRefused::parse(&err),
        Some(LedgerRefused {
            requested: 256,
            cap: total,
            live: total
        }),
        "{err}"
    );
    assert_eq!(exact.live(), total);
    assert_eq!(rt.pool_resident().live_bytes, total);
}

#[test]
fn a_refused_charge_trims_the_free_list_and_retries_once() {
    let (big, small) = (1 << 20, 1);
    let total = bucket(big, BufferKind::Cold);
    let ledger = Fake::new(total);
    let rt = runtime_with(&ledger);

    // The big buffer parks on the free list: still resident, so it still
    // counts against the cap.
    drop(rt.alloc_buffer_kind(big, BufferKind::Cold).unwrap());
    settle(&rt);
    assert_eq!(ledger.cached(), total);
    assert_eq!(ledger.live(), 0);

    // A different bucket cannot reuse it. The charge is refused, the ladder
    // destroys the cached buffer, and the retry succeeds.
    let before = ledger.charges();
    let s = rt.alloc_buffer_kind(small, BufferKind::Cold).unwrap();
    assert_eq!(ledger.cached(), 0);
    assert_eq!(ledger.live(), bucket(small, BufferKind::Cold));
    assert_eq!(ledger.charges() - before, 1, "the refused attempt charged nothing");
    assert_eq!(
        rt.pool_resident(),
        PoolResident {
            live_bytes: bucket(small, BufferKind::Cold),
            cached_bytes: 0
        }
    );

    // The free list is empty now, so a second refusal trims nothing and does
    // not retry: the request is refused and nothing is charged.
    let before = ledger.charges();
    let err = rt.alloc_buffer_kind(big, BufferKind::Cold).map(|_| ()).unwrap_err();
    assert!(LedgerRefused::parse(&err).is_some(), "{err}");
    assert_eq!(ledger.charges(), before);
    drop(s);
    assert_eq!(ledger.unmatched(), 0);
}

#[test]
fn live_bytes_are_never_trimmed() {
    let big = 1 << 20;
    let total = bucket(big, BufferKind::Cold);
    let ledger = Fake::new(total);
    let rt = runtime_with(&ledger);
    let held = rt.alloc_buffer_kind(big, BufferKind::Cold).unwrap();
    let err = rt.alloc_buffer_kind(1, BufferKind::Cold).map(|_| ()).unwrap_err();
    assert!(LedgerRefused::parse(&err).is_some(), "{err}");
    assert_eq!(ledger.live(), total, "the live buffer kept its charge");
    drop(held);
}

#[test]
fn a_free_list_hit_needs_no_new_room() {
    let n = 1 << 20;
    let total = bucket(n, BufferKind::Cold);
    let ledger = Fake::new(total);
    let rt = runtime_with(&ledger);
    let first = rt.alloc_buffer_kind(n, BufferKind::Cold).unwrap();
    let at = addr(&first);
    drop(first);
    settle(&rt);
    assert_eq!((ledger.live(), ledger.cached()), (0, total));

    let again = rt.alloc_buffer_kind(n, BufferKind::Cold).unwrap();
    assert_eq!(addr(&again), at, "served from the free list");
    assert_eq!((ledger.live(), ledger.cached()), (total, 0));
    assert_eq!(ledger.unmatched(), 0);
}

#[test]
fn a_ledger_cannot_be_installed_over_live_buffers() {
    let rt = GpuRuntime::new().unwrap();
    let held = rt.alloc_buffer(1 << 20).unwrap();
    let ledger: Arc<dyn PoolLedger> = Fake::new(u64::MAX);
    let err = rt.set_pool_ledger(Some(ledger.clone())).unwrap_err();
    assert!(err.contains("before the first pool allocation"), "{err}");
    drop(held);
    // Parked on the free list it is still a held buffer without a charge.
    settle(&rt);
    assert!(rt.set_pool_ledger(Some(ledger)).is_err());
}

#[test]
fn without_a_ledger_the_pool_only_counts() {
    let rt = GpuRuntime::new().unwrap();
    let n = 1 << 20;
    let first = rt.alloc_buffer(n).unwrap();
    let at = addr(&first);
    assert_eq!(rt.pool_resident().live_bytes, bucket(n, BufferKind::Cold));
    drop(first);
    settle(&rt);
    assert_eq!(rt.pool_resident().cached_bytes, bucket(n, BufferKind::Cold));
    let again = rt.alloc_buffer(n).unwrap();
    assert_eq!(addr(&again), at);
    assert_eq!(rt.pool_resident().cached_bytes, 0);
    assert_eq!(rt.pool_resident().live_bytes, bucket(n, BufferKind::Cold));
}

#[test]
fn setting_the_cache_cap_releases_what_it_trims() {
    let ledger = Fake::new(u64::MAX);
    let rt = runtime_with(&ledger);
    let sizes = [1 << 20, 2 << 20, 3 << 20];
    let bufs: Vec<_> = sizes.iter().map(|&n| rt.alloc_buffer(n).unwrap()).collect();
    drop(bufs);
    settle(&rt);
    let all: u64 = sizes.iter().map(|&n| bucket(n, BufferKind::Cold)).sum();
    assert_eq!(ledger.cached(), all);
    rt.set_pool_cache_cap_bytes(0);
    assert_eq!(ledger.cached(), 0);
    assert_eq!(rt.pool_resident().cached_bytes, 0);
    assert_eq!(ledger.unmatched(), 0);
}

#[test]
fn a_buffer_over_the_cache_cap_is_destroyed_and_released() {
    let ledger = Fake::new(u64::MAX);
    let rt = runtime_with(&ledger);
    rt.set_pool_cache_cap_bytes(1 << 20);
    drop(rt.alloc_buffer(2 << 20).unwrap());
    settle(&rt);
    assert_eq!((ledger.live(), ledger.cached()), (0, 0));
    assert_eq!(rt.pool_resident(), PoolResident::default());
}

#[test]
fn dropping_the_runtime_returns_every_charge() {
    let ledger = Fake::new(u64::MAX);
    let rt = runtime_with(&ledger);
    // Cached, waiting on the GPU (cold and hot), and still held.
    drop(rt.alloc_buffer(1 << 20).unwrap());
    settle(&rt);
    // A commit that has not finished: nothing dropped from here on is released.
    rt.metal4.event_value.fetch_add(1, Ordering::AcqRel);
    drop(rt.alloc_buffer(2 << 20).unwrap());
    drop(rt.alloc_buffer_hot(3 << 20).unwrap());
    let held = rt.alloc_buffer(5 << 20).unwrap();
    assert!(rt.deferred_pending());
    assert_eq!(ledger.cached(), bucket(1 << 20, BufferKind::Cold));
    assert_eq!(
        ledger.live(),
        bucket(2 << 20, BufferKind::Cold) + bucket(3 << 20, BufferKind::Hot) + bucket(5 << 20, BufferKind::Cold)
    );

    drop(rt);
    assert_eq!(ledger.cached(), 0);
    assert_eq!(
        ledger.live(),
        bucket(5 << 20, BufferKind::Cold),
        "only the buffer that outlives the runtime is still charged"
    );
    drop(held);
    assert_eq!(ledger.live(), 0);
    assert_eq!(ledger.unmatched(), 0);
}

/// Make the commit after `rt`'s current one submitted and finished, so a
/// buffer dropped before it may be released.
fn finish_next_commit(rt: &GpuRuntime) -> u64 {
    let next = rt.metal4.event_value.fetch_add(1, Ordering::AcqRel) + 1;
    rt.metal4.lifetime_event.setSignaledValue(next);
    next
}

#[test]
fn reclaim_frees_only_what_the_gpu_finished_and_never_waits() {
    let ledger = Fake::new(u64::MAX);
    let rt = runtime_with(&ledger);
    let n = 1 << 20;
    drop(rt.alloc_buffer(n).unwrap());
    assert_eq!(
        ledger.live(),
        bucket(n, BufferKind::Cold),
        "waiting for the GPU: still live"
    );

    // Nothing finished yet: reclaiming up to any value frees nothing.
    assert_eq!(rt.reclaim(u64::MAX).unwrap(), 0);
    assert_eq!(ledger.live(), bucket(n, BufferKind::Cold));

    let done = finish_next_commit(&rt);
    // The caller names a value below the buffer's release point.
    assert_eq!(rt.reclaim(done - 1).unwrap(), 0);
    assert_eq!(ledger.live(), bucket(n, BufferKind::Cold));
    // A value above what the lifetime event reached is clamped to it.
    assert_eq!(rt.reclaim(u64::MAX).unwrap(), bucket(n, BufferKind::Cold));
    assert_eq!((ledger.live(), ledger.cached()), (0, bucket(n, BufferKind::Cold)));
    assert_eq!(ledger.unmatched(), 0);
}

#[test]
fn reclaim_refuses_a_poisoned_runtime() {
    let rt = GpuRuntime::new().unwrap();
    rt.poison.set("test");
    assert!(rt.reclaim(0).is_err());
}
