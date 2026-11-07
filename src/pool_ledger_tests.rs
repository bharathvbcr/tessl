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

/// Panic while holding the pool lock, as a ledger callback that panics would.
fn poison_pool(rt: &GpuRuntime) {
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _held = rt.pool.lock().unwrap();
        panic!("poisoning the pool on purpose");
    }));
    assert!(unwound.is_err());
    assert!(rt.pool.is_poisoned());
}

/// A drop that is drained after the pool lock was poisoned still hands its
/// live charge back: it is parked (Cold) or destroyed (Hot), never forgotten.
#[test]
fn a_poisoned_pool_still_releases_the_live_charge_of_a_drained_drop() {
    let ledger = Fake::new(u64::MAX);
    let rt = runtime_with(&ledger);
    rt.synchronize().unwrap();
    let cold = rt.alloc_buffer(300_000).unwrap();
    let hot = rt.alloc_buffer_hot(3 << 20).unwrap();
    let (cold_key, hot_key) = (bucket(300_000, BufferKind::Cold), bucket(3 << 20, BufferKind::Hot));
    assert_eq!(ledger.live(), cold_key + hot_key);

    poison_pool(&rt);
    drop(cold);
    drop(hot);
    settle(&rt);

    assert_eq!(ledger.live(), 0, "the Live charge of a drained drop was never released");
    assert_eq!(
        ledger.cached(),
        cold_key,
        "the Cold buffer was not parked on the free list"
    );
    assert_eq!(ledger.unmatched(), 0);
    let held = rt.pool_resident();
    assert_eq!((held.live_bytes, held.cached_bytes), (0, cold_key));
}

/// The pool's other release paths do not skip a poisoned lock either: the
/// cache cap trims and `trim_free_list` frees, each releasing its Cached charge.
#[test]
fn a_poisoned_pool_still_trims_its_free_list() {
    let ledger = Fake::new(u64::MAX);
    let rt = runtime_with(&ledger);
    rt.synchronize().unwrap();
    let key = bucket(300_000, BufferKind::Cold);
    drop((rt.alloc_buffer(300_000).unwrap(), rt.alloc_buffer(300_000).unwrap()));
    settle(&rt);
    assert_eq!(ledger.cached(), 2 * key);

    poison_pool(&rt);
    rt.set_pool_cache_cap_bytes(key as usize);
    assert_eq!(ledger.cached(), key, "a poisoned lock skipped the cache cap");
    assert_eq!(rt.trim_free_list(), key as usize);
    assert_eq!((ledger.live(), ledger.cached(), ledger.unmatched()), (0, 0, 0));
}

// ---- adversarial: a ledger that panics on its Nth call ----

/// Delegates to a `Fake`, panicking on the Nth call (counted from `arm`) of
/// any ledger method, before the inner ledger sees it.
struct Flaky {
    inner: Arc<Fake>,
    countdown: std::sync::atomic::AtomicI64,
    only_release: bool,
}

impl Flaky {
    fn new(inner: &Arc<Fake>, only_release: bool) -> Arc<Self> {
        Arc::new(Self {
            inner: inner.clone(),
            countdown: std::sync::atomic::AtomicI64::new(i64::MAX),
            only_release,
        })
    }
    fn arm(&self, nth: i64) {
        self.countdown.store(nth, Ordering::SeqCst);
    }
    fn disarm(&self) {
        self.countdown.store(i64::MAX, Ordering::SeqCst);
    }
    fn tick(&self) {
        if self.countdown.fetch_sub(1, Ordering::SeqCst) == 1 {
            panic!("flaky ledger: injected panic");
        }
    }
}

impl PoolLedger for Flaky {
    fn try_charge(&self, class: LedgerClass, bytes: u64) -> Result<(), LedgerRefused> {
        if !self.only_release {
            self.tick();
        }
        self.inner.try_charge(class, bytes)
    }
    fn release(&self, class: LedgerClass, bytes: u64) {
        self.tick();
        self.inner.release(class, bytes)
    }
}

fn freelist_bytes(rt: &GpuRuntime) -> u64 {
    let pool = rt.pool.lock().unwrap_or_else(PoisonError::into_inner);
    pool.freelist.iter().map(|(k, v)| (*k * v.len()) as u64).sum()
}

/// After a ledger panic at call `nth` of `op`, the recovered pool must never
/// release a charge twice (`unmatched`), and the ledger's Cached balance must
/// cover what the free list holds (no under-charge a later trim would trip).
fn flaky_run(nth: i64, only_release: bool, op: impl Fn(&GpuRuntime)) -> (u64, u64, u64) {
    let fake = Fake::new(u64::MAX);
    let flaky = Flaky::new(&fake, only_release);
    let rt = GpuRuntime::new().unwrap();
    let as_dyn: Arc<dyn PoolLedger> = flaky.clone();
    rt.set_pool_ledger(Some(as_dyn)).unwrap();
    rt.synchronize().unwrap();
    // Three distinct cached buckets, then a few live buffers to drain in op.
    drop((
        rt.alloc_buffer(300_000).unwrap(),
        rt.alloc_buffer(300_000).unwrap(),
        rt.alloc_buffer(1 << 20).unwrap(),
    ));
    settle(&rt);
    let live = vec![rt.alloc_buffer(5_000).unwrap(), rt.alloc_buffer_hot(3 << 20).unwrap()];
    drop(live);
    flaky.arm(nth);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| op(&rt)));
    flaky.disarm();
    assert!(
        fake.cached() >= freelist_bytes(&rt),
        "nth={nth}: ledger cached {} below free list {}",
        fake.cached(),
        freelist_bytes(&rt)
    );
    // Recovery paths: trim, drain, resident, then Drop.
    let _ = rt.trim_free_list();
    settle(&rt);
    let _ = rt.pool_resident();
    assert_eq!(
        fake.unmatched(),
        0,
        "nth={nth}: a charge was released twice after recovery"
    );
    drop(rt);
    assert_eq!(fake.unmatched(), 0, "nth={nth}: Drop released a charge twice");
    (fake.live(), fake.cached(), nth as u64)
}

#[test]
fn adversarial_panic_in_trim_never_double_releases() {
    for nth in 1..=6 {
        let (live, cached, _) = flaky_run(nth, true, |rt| {
            rt.trim_free_list();
        });
        assert_eq!(
            (live, cached),
            (0, 0),
            "trim nth={nth}: charges left on the ledger after Drop"
        );
    }
}

#[test]
fn adversarial_panic_in_cache_cap_trim_never_double_releases() {
    for nth in 1..=3 {
        let (live, cached, _) = flaky_run(nth, true, |rt| rt.set_pool_cache_cap_bytes(256));
        assert_eq!(
            (live, cached),
            (0, 0),
            "cap nth={nth}: charges left on the ledger after Drop"
        );
    }
}

#[test]
fn adversarial_panic_in_drain_never_double_releases() {
    for nth in 1..=12 {
        let (live, cached, _) = flaky_run(nth, false, |rt| {
            rt.set_pool_cache_cap_bytes(400_000);
            settle(rt);
        });
        assert_eq!(
            (live, cached),
            (0, 0),
            "drain nth={nth}: charges left on the ledger after Drop"
        );
    }
}

/// A free-list hit moves a buffer's charge from Cached to Live. A panic in
/// either ledger call must leave the books balanced once the runtime is gone.
#[test]
fn adversarial_panic_in_free_list_hit_never_double_releases() {
    for nth in 1..=4 {
        let (live, cached, _) = flaky_run(nth, false, |rt| {
            let _hit = rt.alloc_buffer(300_000);
        });
        assert_eq!(
            (live, cached),
            (0, 0),
            "hit nth={nth}: charges left on the ledger after Drop"
        );
    }
}

/// `try_lock` on a lock this thread holds is WouldBlock even when poisoned
/// earlier: the recursive call returns 0 rather than recovering or deadlocking.
#[test]
fn adversarial_trim_while_this_thread_holds_a_poisoned_pool_returns_zero() {
    let ledger = Fake::new(u64::MAX);
    let rt = runtime_with(&ledger);
    rt.synchronize().unwrap();
    drop(rt.alloc_buffer(300_000).unwrap());
    settle(&rt);
    poison_pool(&rt);
    let guard = rt.pool_recovering();
    assert_eq!(rt.trim_free_list(), 0);
    drop(guard);
    assert!(rt.trim_free_list() > 0);
}

/// Drop with a poisoned pool hands back every charge exactly once.
#[test]
fn adversarial_drop_with_poisoned_pool_releases_everything_once() {
    let ledger = Fake::new(u64::MAX);
    let rt = runtime_with(&ledger);
    rt.synchronize().unwrap();
    drop((rt.alloc_buffer(300_000).unwrap(), rt.alloc_buffer(1 << 20).unwrap()));
    settle(&rt);
    let held = rt.alloc_buffer(5_000).unwrap();
    drop(held);
    poison_pool(&rt);
    // Alloc refuses a poisoned pool (unchanged behavior).
    assert!(rt.alloc_buffer(5_000).is_err());
    drop(rt);
    assert_eq!((ledger.live(), ledger.cached(), ledger.unmatched()), (0, 0, 0));
}
