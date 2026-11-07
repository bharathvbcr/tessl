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

// ---- adversarial pass 2: sweeps over every entry, refusals, guards ----

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Charge,
    Release,
}

/// A ledger that panics on the Nth call (optionally only calls of one kind and
/// class), on every Kth call, or refuses the Nth `try_charge`. A panic or a
/// refusal comes before the inner `Fake` sees the call, which is the contract
/// the pool relies on (a call that unwinds has changed nothing).
struct Chaos {
    fake: Arc<Fake>,
    filter: Mutex<Option<(Kind, LedgerClass)>>,
    seen: std::sync::atomic::AtomicI64,
    panic_at: std::sync::atomic::AtomicI64,
    every: std::sync::atomic::AtomicI64,
    refuse_at: std::sync::atomic::AtomicI64,
    refuse_all: AtomicBool,
    charges: std::sync::atomic::AtomicI64,
    on_release: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
}

impl Chaos {
    fn new(fake: &Arc<Fake>) -> Arc<Self> {
        Arc::new(Self {
            fake: fake.clone(),
            filter: Mutex::new(None),
            seen: Default::default(),
            panic_at: Default::default(),
            every: Default::default(),
            refuse_at: Default::default(),
            refuse_all: AtomicBool::new(false),
            charges: Default::default(),
            on_release: Mutex::new(None),
        })
    }
    fn reset(&self) {
        for c in [&self.seen, &self.panic_at, &self.every, &self.refuse_at, &self.charges] {
            c.store(0, Ordering::SeqCst);
        }
        self.refuse_all.store(false, Ordering::SeqCst);
        *self.filter.lock().unwrap() = None;
    }
    fn panic_on_call(&self, nth: i64) {
        self.reset();
        self.panic_at.store(nth, Ordering::SeqCst);
    }
    fn panic_on_kind(&self, kind: Kind, class: LedgerClass, nth: i64) {
        self.panic_on_call(nth);
        *self.filter.lock().unwrap() = Some((kind, class));
    }
    fn panic_every(&self, k: i64) {
        self.reset();
        self.every.store(k, Ordering::SeqCst);
    }
    fn refuse_nth_charge(&self, nth: i64) {
        self.reset();
        self.refuse_at.store(nth, Ordering::SeqCst);
    }
    fn tick(&self, kind: Kind, class: LedgerClass) {
        let hit = self
            .filter
            .lock()
            .unwrap()
            .is_none_or(|(k, c)| k == kind && std::mem::discriminant(&c) == std::mem::discriminant(&class));
        if !hit {
            return;
        }
        let n = self.seen.fetch_add(1, Ordering::SeqCst) + 1;
        let (at, every) = (self.panic_at.load(Ordering::SeqCst), self.every.load(Ordering::SeqCst));
        if (at != 0 && n == at) || (every > 0 && n % every == 0) {
            panic!("chaos ledger: injected panic at call {n}");
        }
    }
}

impl PoolLedger for Chaos {
    fn try_charge(&self, class: LedgerClass, bytes: u64) -> Result<(), LedgerRefused> {
        self.tick(Kind::Charge, class);
        let n = self.charges.fetch_add(1, Ordering::SeqCst) + 1;
        if self.refuse_all.load(Ordering::SeqCst) || n == self.refuse_at.load(Ordering::SeqCst) {
            return Err(LedgerRefused {
                requested: bytes,
                cap: 0,
                live: 0,
            });
        }
        self.fake.try_charge(class, bytes)
    }
    fn release(&self, class: LedgerClass, bytes: u64) {
        if let Some(f) = &*self.on_release.lock().unwrap_or_else(PoisonError::into_inner) {
            f();
        }
        self.tick(Kind::Release, class);
        self.fake.release(class, bytes)
    }
}

/// Swallows the injected panics' messages and passes every other panic on.
struct QuietHook(Arc<Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send + 'static>>);

fn quiet_injected() -> QuietHook {
    let prev = Arc::new(std::panic::take_hook());
    let passed = prev.clone();
    std::panic::set_hook(Box::new(move |info| {
        let msg = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| info.payload().downcast_ref::<&str>().copied())
            .unwrap_or("");
        if !msg.contains("injected panic") {
            passed(info);
        }
    }));
    QuietHook(prev)
}

impl Drop for QuietHook {
    fn drop(&mut self) {
        let prev = self.0.clone();
        std::panic::set_hook(Box::new(move |info| prev(info)));
    }
}

fn buf_addr(b: &ProtocolObject<dyn MTLBuffer>) -> usize {
    b.contents().as_ptr() as usize
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Hit,
    Miss,
    DropAll,
    Settle,
    Cap(usize),
    Evict,
    Trim,
    Sync,
    Reclaim,
}

/// Every entry that touches the ledger, in an order that makes the cap evict,
/// a hit follow a trim, and a drain meet cold and hot items.
const SCENARIO: [Op; 19] = [
    Op::Evict,
    Op::DropAll,
    Op::Settle,
    Op::Hit,
    Op::Miss,
    Op::Hit,
    Op::DropAll,
    Op::Sync,
    Op::Cap(400_000),
    Op::Hit,
    Op::Miss,
    Op::DropAll,
    Op::Reclaim,
    Op::Trim,
    Op::Cap(usize::MAX),
    Op::Hit,
    Op::DropAll,
    Op::Settle,
    Op::Trim,
];

struct Rig {
    held: Vec<crate::tensor::GpuBuffer>,
    rt: Arc<GpuRuntime>,
    fake: Arc<Fake>,
    chaos: Arc<Chaos>,
}

impl Rig {
    fn new() -> Self {
        let fake = Fake::new(u64::MAX);
        let chaos = Chaos::new(&fake);
        let rt = GpuRuntime::new().unwrap();
        let as_dyn: Arc<dyn PoolLedger> = chaos.clone();
        rt.set_pool_ledger(Some(as_dyn)).unwrap();
        rt.synchronize().unwrap();
        Self {
            held: Vec::new(),
            rt,
            fake,
            chaos,
        }
    }

    /// Three cached buffers (two buckets) and three held ones (cold, hot, cold).
    fn seed(&mut self) {
        let parked = (
            self.rt.alloc_buffer(300_000).unwrap(),
            self.rt.alloc_buffer(300_000).unwrap(),
            self.rt.alloc_buffer(1 << 20).unwrap(),
        );
        drop(parked);
        settle(&self.rt);
        self.held.push(self.rt.alloc_buffer(5_000).unwrap());
        self.held.push(self.rt.alloc_buffer_hot(3 << 20).unwrap());
        self.held.push(self.rt.alloc_buffer(700_000).unwrap());
    }

    fn apply(&mut self, op: Op) {
        match op {
            Op::Hit => {
                if let Ok(b) = self.rt.alloc_buffer(300_000) {
                    self.held.push(b);
                }
            }
            Op::Miss => {
                if let Ok(b) = self.rt.alloc_buffer(2 << 20) {
                    self.held.push(b);
                }
                if let Ok(b) = self.rt.alloc_buffer_hot(100_000) {
                    self.held.push(b);
                }
            }
            Op::DropAll => self.held.clear(),
            Op::Settle => {
                let _ = self.rt.recycle_if_idle();
            }
            Op::Cap(n) => self.rt.set_pool_cache_cap_bytes(n),
            Op::Evict => self
                .rt
                .set_pool_cache_cap_bytes(self.rt.pool_resident().cached_bytes as usize),
            Op::Trim => {
                self.rt.trim_free_list();
            }
            Op::Sync => {
                let _ = self.rt.synchronize();
            }
            Op::Reclaim => {
                let _ = self.rt.reclaim(u64::MAX);
            }
        }
    }

    /// The pool's counters, its lists and queues, and the ledger's balances all
    /// say the same thing, and no buffer is in two places.
    fn audit(&self, ctx: &str) {
        let r = self.rt.pool_resident();
        let (live, cached) = (self.fake.live(), self.fake.cached());
        assert_eq!(
            (r.live_bytes, r.cached_bytes),
            (live, cached),
            "{ctx}: pool_resident disagrees with the ledger"
        );
        assert_eq!(self.fake.unmatched(), 0, "{ctx}: an unmatched release");
        let mut seen = std::collections::HashSet::new();
        let (mut listed, mut queued, mut held) = (0u64, 0u64, 0u64);
        {
            let pool = self.rt.pool_recovering();
            for (key, buffers) in &pool.freelist {
                for b in buffers {
                    assert_eq!(b.length(), *key, "{ctx}: a buffer on the wrong list");
                    assert!(seen.insert(buf_addr(b)), "{ctx}: a buffer listed twice");
                    listed += *key as u64;
                }
            }
        }
        for (_, (b, _)) in self
            .rt
            .pending_cold_recycle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
        {
            assert!(seen.insert(buf_addr(b)), "{ctx}: a cold item queued twice");
            queued += b.length() as u64;
        }
        for (_, b) in self
            .rt
            .pending_retirement
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
        {
            assert!(seen.insert(buf_addr(b)), "{ctx}: a hot item queued twice");
            queued += b.length() as u64;
        }
        for b in &self.held {
            assert!(seen.insert(addr(b)), "{ctx}: a held buffer also listed or queued");
            held += b.metal().length() as u64;
        }
        assert_eq!(listed, cached, "{ctx}: free list bytes vs ledger Cached");
        assert_eq!(queued + held, live, "{ctx}: queued + held bytes vs ledger Live");
    }

    /// Disarm, let everything drain, and drop in one of two orders. Every
    /// charge must be back.
    fn finish(mut self, runtime_first: bool, ctx: &str) {
        self.chaos.reset();
        self.apply(Op::Settle);
        self.apply(Op::Reclaim);
        self.audit(ctx);
        let Rig { held, rt, fake, chaos } = self;
        if runtime_first {
            drop(rt);
            drop(held);
        } else {
            drop(held);
            settle(&rt);
            assert!(!rt.deferred_pending(), "{ctx}: the drain left items queued");
            drop(rt);
        }
        drop(chaos);
        assert_eq!(
            (fake.live(), fake.cached(), fake.unmatched()),
            (0, 0, 0),
            "{ctx}: charges left on the ledger after Drop"
        );
    }
}

fn poison_queue<T>(queue: &Mutex<Vec<Deferred<T>>>) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _held = queue.lock().unwrap();
        panic!("injected panic: poisoning a queue on purpose");
    }));
    assert!(queue.is_poisoned());
}

#[derive(Clone, Copy, Debug)]
enum Fault {
    PanicNth(i64),
    PanicEvery(i64),
    RefuseNth(i64),
    RefuseNthPanicNth(i64, i64),
}

fn run_scenario(fault: Fault, clear_poison: bool, poison_first: bool) {
    let mut rig = Rig::new();
    rig.seed();
    rig.audit("seeded");
    if poison_first {
        poison_pool(&rig.rt);
        poison_queue(&rig.rt.pending_cold_recycle);
        poison_queue(&rig.rt.pending_retirement);
    }
    match fault {
        Fault::PanicNth(n) => rig.chaos.panic_on_call(n),
        Fault::PanicEvery(k) => rig.chaos.panic_every(k),
        Fault::RefuseNth(k) => rig.chaos.refuse_nth_charge(k),
        Fault::RefuseNthPanicNth(k, n) => {
            rig.chaos.panic_on_call(n);
            rig.chaos.refuse_at.store(k, Ordering::SeqCst);
        }
    }
    for (i, op) in SCENARIO.into_iter().enumerate() {
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rig.apply(op))).is_err();
        if unwound && clear_poison {
            rig.rt.pool.clear_poison();
        }
        rig.audit(&format!(
            "{fault:?} clear={clear_poison} poisoned={poison_first} op#{i} {op:?} unwound={unwound}"
        ));
    }
    let ctx = format!("{fault:?} clear={clear_poison} poisoned={poison_first} finish");
    rig.finish(
        matches!(fault, Fault::PanicNth(n) | Fault::RefuseNthPanicNth(_, n) if n % 2 == 0),
        &ctx,
    );
}

#[test]
fn sweep_a_panic_on_the_nth_ledger_call_in_every_entry() {
    let _quiet = quiet_injected();
    for n in 1..=50 {
        run_scenario(Fault::PanicNth(n), false, false);
    }
}

#[test]
fn sweep_the_nth_panic_with_the_poison_cleared_between_ops() {
    let _quiet = quiet_injected();
    for n in 1..=50 {
        run_scenario(Fault::PanicNth(n), true, false);
    }
}

#[test]
fn sweep_the_nth_panic_with_the_pool_and_both_queues_poisoned_first() {
    let _quiet = quiet_injected();
    for n in 1..=50 {
        run_scenario(Fault::PanicNth(n), false, true);
    }
}

#[test]
fn sweep_a_panic_on_every_kth_call_so_one_run_unwinds_many_times() {
    let _quiet = quiet_injected();
    for k in 1..=9 {
        run_scenario(Fault::PanicEvery(k), true, false);
    }
}

#[test]
fn sweep_a_refusal_at_every_try_charge_keeps_the_books_balanced() {
    for k in 1..=40 {
        run_scenario(Fault::RefuseNth(k), false, false);
    }
}

#[test]
fn sweep_a_refusal_and_a_panic_in_the_same_run() {
    let _quiet = quiet_injected();
    for k in [1, 2, 3, 5, 8] {
        for n in 1..=15 {
            run_scenario(Fault::RefuseNthPanicNth(k, n), true, false);
        }
    }
}

/// The Cached charge at the end of a drained drop panics: the Live charge is
/// already given back, so the buffer is destroyed, not queued again.
#[test]
fn a_panic_in_the_cached_charge_of_a_drained_drop_destroys_the_buffer_once() {
    let _quiet = quiet_injected();
    let rig = Rig::new();
    let buf = rig.rt.alloc_buffer(300_000).unwrap();
    let key = bucket(300_000, BufferKind::Cold);
    assert_eq!(rig.fake.live(), key);
    drop(buf);
    rig.chaos.panic_on_kind(Kind::Charge, LedgerClass::Cached, 1);
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| settle(&rig.rt))).is_err();
    assert!(unwound, "the Cached charge was never reached");
    rig.audit("after the panic");
    assert_eq!((rig.fake.live(), rig.fake.cached()), (0, 0));
    assert!(!rig.rt.deferred_pending(), "a released buffer was queued again");
    rig.chaos.reset();
    settle(&rig.rt);
    rig.audit("after a second drain");
    rig.finish(false, "cached charge panic");
}

/// A free-list hit whose Live charge panics: the Cached charge is already
/// released and the buffer is gone, nothing is left to release twice.
#[test]
fn a_panic_in_the_live_charge_of_a_hit_loses_the_buffer_and_nothing_else() {
    let _quiet = quiet_injected();
    let mut rig = Rig::new();
    rig.seed();
    let key = bucket(300_000, BufferKind::Cold);
    let cached_before = rig.fake.cached();
    rig.chaos.panic_on_kind(Kind::Charge, LedgerClass::Live, 1);
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rig.apply(Op::Hit))).is_err();
    assert!(unwound);
    rig.audit("after the panic");
    assert_eq!(
        rig.fake.cached(),
        cached_before - key,
        "one cached buffer was destroyed"
    );
    rig.chaos.reset();
    rig.rt.pool.clear_poison();
    rig.apply(Op::Hit);
    rig.audit("a later hit");
    rig.finish(false, "live charge panic in a hit");
}

/// A refusal of the Cached charge in a drained drop destroys the buffer.
#[test]
fn a_refused_cached_charge_destroys_the_drained_buffer_and_the_books_agree() {
    let rig = Rig::new();
    let buf = rig.rt.alloc_buffer(300_000).unwrap();
    drop(buf);
    rig.chaos.refuse_nth_charge(1);
    settle(&rig.rt);
    rig.audit("after the refusal");
    assert_eq!((rig.fake.live(), rig.fake.cached()), (0, 0));
    assert_eq!(rig.rt.pool_resident(), PoolResident::default());
    assert!(!rig.rt.deferred_pending());
    rig.finish(false, "refused cached charge");
}

/// A refusal of the Live charge of a hit destroys the cached buffer, and the
/// request is served as a fresh one.
#[test]
fn a_refused_live_charge_on_a_hit_destroys_the_cached_buffer_and_serves_a_fresh_one() {
    let mut rig = Rig::new();
    rig.seed();
    let key = bucket(300_000, BufferKind::Cold);
    let cached_before = rig.fake.cached();
    rig.chaos.refuse_nth_charge(1);
    rig.apply(Op::Hit);
    assert_eq!(rig.held.len(), 4, "the request was served");
    rig.audit("after the refused hit");
    assert_eq!(rig.fake.cached(), cached_before - key);
    rig.finish(true, "refused live charge on a hit");
}

/// Every charge refused: the ladder trims the free list, retries, refuses; the
/// request fails and what was trimmed was released exactly once.
#[test]
fn a_ledger_that_refuses_everything_leaves_the_books_balanced() {
    let mut rig = Rig::new();
    rig.seed();
    rig.chaos.reset();
    rig.chaos.refuse_all.store(true, Ordering::SeqCst);
    for n in [300_000, 2 << 20, 5_000] {
        let err = rig.rt.alloc_buffer(n).map(|_| ()).unwrap_err();
        assert!(LedgerRefused::parse(&err).is_some(), "{err}");
        rig.audit(&format!("refused alloc {n}"));
    }
    assert_eq!(rig.fake.cached(), 0, "the ladder trimmed the free list");
    rig.apply(Op::DropAll);
    rig.apply(Op::Settle);
    rig.audit("held drops drained under a refusing ledger");
    assert_eq!(rig.fake.live(), 0);
    rig.finish(false, "refuse all");
}

/// A release that calls back into the runtime's `try_lock` entry from inside
/// the pool lock. The base (f802c6b) also called the ledger under the pool
/// lock in alloc, recycle and the trims, so this is not new reentrancy; the
/// `try_lock` path returns 0 rather than deadlocking.
#[test]
fn a_release_that_calls_trim_free_list_from_inside_the_pool_lock_does_not_deadlock() {
    let mut rig = Rig::new();
    rig.seed();
    let rt_addr = Arc::as_ptr(&rig.rt) as usize;
    let calls = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let freed = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let (c, f) = (calls.clone(), freed.clone());
    *rig.chaos.on_release.lock().unwrap() = Some(Box::new(move || {
        // SAFETY: the callback runs on the thread that owns the runtime, while
        // a `&GpuRuntime` call is on the stack, so the pointee is alive.
        let rt = unsafe { &*(rt_addr as *const GpuRuntime) };
        f.fetch_add(rt.trim_free_list() as u64, Ordering::SeqCst);
        c.fetch_add(1, Ordering::SeqCst);
    }));
    for op in SCENARIO {
        rig.apply(op);
        rig.audit(&format!("reentrant {op:?}"));
    }
    assert!(calls.load(Ordering::SeqCst) > 0, "the callback never ran");
    assert_eq!(freed.load(Ordering::SeqCst), 0, "a reentrant trim got the lock");
    *rig.chaos.on_release.lock().unwrap() = None;
    rig.finish(false, "reentrant trim");
}

/// A panic mid-drain requeues items whose residency was already removed; the
/// retry removes it again. The residency set tolerates that, but its count
/// must too: one add, one remove.
#[test]
#[ignore = "BREAK: a requeued drain item is unregistered twice, so residency_count drifts below the set's size"]
fn a_requeued_drain_item_leaves_the_residency_count_where_it_started() {
    let _quiet = quiet_injected();
    let mut rig = Rig::new();
    rig.seed();
    let count = || rig.rt.metal4.residency_count.load(Ordering::Relaxed);
    let before = count();
    let extra: Vec<_> = (0..4).map(|_| rig.rt.alloc_buffer(300_000).unwrap()).collect();
    assert_eq!(count(), before + 4);
    drop(extra);
    rig.chaos.panic_on_kind(Kind::Release, LedgerClass::Live, 1);
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| settle(&rig.rt))).is_err();
    assert!(unwound);
    rig.chaos.reset();
    settle(&rig.rt);
    assert_eq!(
        count(),
        before,
        "four buffers were added and removed once each, but were unregistered twice"
    );
}

/// The claim "an unwinding panic leaves the rest in place for the next trim,
/// drain or Drop" has no next Drop when the panic is in Drop itself: the
/// remaining free list and queues are destroyed with their charges held.
#[test]
#[ignore = "BREAK: a ledger panic inside GpuRuntime::drop leaks every charge after it"]
fn a_ledger_panic_inside_drop_still_returns_the_remaining_charges() {
    let _quiet = quiet_injected();
    for n in 1..=8 {
        let mut rig = Rig::new();
        rig.seed();
        rig.apply(Op::DropAll);
        let Rig { rt, fake, chaos, held } = rig;
        drop(held);
        chaos.panic_on_call(n);
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(rt))).is_err();
        assert!(unwound, "Drop made fewer than {n} ledger calls");
        assert_eq!(
            (fake.live(), fake.cached()),
            (0, 0),
            "a panic at ledger call {n} inside Drop left charges on the ledger"
        );
    }
}

/// A seeded random op sequence with no faults. The same test runs on the base
/// (f802c6b) and the output is diffed: the pool's balances after each op must
/// be identical, and match the ledger's, with and without a binding cap.
#[test]
fn differential_balance_trace() {
    for cap in [u64::MAX, 6 << 20] {
        let fake = Fake::new(cap);
        let rt = runtime_with(&fake);
        rt.synchronize().unwrap();
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let sizes = [1usize, 5_000, 70_000, 300_000, 1 << 20, (1 << 20) + 1, 3 << 20];
        let mut held: Vec<crate::tensor::GpuBuffer> = Vec::new();
        for step in 0..300 {
            let r = next();
            let what = match r % 10 {
                0..=3 => {
                    let n = sizes[(next() % sizes.len() as u64) as usize];
                    let hot = next() % 3 == 0;
                    let got = if hot {
                        rt.alloc_buffer_hot(n)
                    } else {
                        rt.alloc_buffer(n)
                    };
                    let ok = got.is_ok();
                    held.extend(got);
                    format!("alloc {n} hot={hot} ok={ok}")
                }
                4 | 5 => {
                    let k = (next() % 4) as usize;
                    for _ in 0..k.min(held.len()) {
                        let i = (next() % held.len() as u64) as usize;
                        held.swap_remove(i);
                    }
                    format!("drop {k}")
                }
                6 => {
                    settle(&rt);
                    "settle".to_string()
                }
                7 => {
                    let caps = [0usize, 1 << 20, 4 << 20, usize::MAX];
                    let c = caps[(next() % 4) as usize];
                    rt.set_pool_cache_cap_bytes(c);
                    format!("cap {c}")
                }
                8 => format!("trim {}", rt.trim_free_list()),
                _ => {
                    held.clear();
                    settle(&rt);
                    "dropall settle".to_string()
                }
            };
            let res = rt.pool_resident();
            println!(
                "TRACE cap={cap} {step} {what} live={} cached={} resident=({},{})",
                fake.live(),
                fake.cached(),
                res.live_bytes,
                res.cached_bytes
            );
            assert_eq!((res.live_bytes, res.cached_bytes), (fake.live(), fake.cached()));
            assert_eq!(fake.unmatched(), 0);
        }
        drop(held);
        settle(&rt);
        drop(rt);
        assert_eq!((fake.live(), fake.cached(), fake.unmatched()), (0, 0, 0));
    }
}
