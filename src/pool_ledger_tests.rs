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
    /// The byte count of the call the last injected panic came from.
    panicked_bytes: AtomicU64,
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
            panicked_bytes: AtomicU64::new(0),
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
    fn tick(&self, kind: Kind, class: LedgerClass, bytes: u64) {
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
            self.panicked_bytes.store(bytes, Ordering::SeqCst);
            panic!("chaos ledger: injected panic at call {n}");
        }
    }
}

impl PoolLedger for Chaos {
    fn try_charge(&self, class: LedgerClass, bytes: u64) -> Result<(), LedgerRefused> {
        self.tick(Kind::Charge, class, bytes);
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
        self.tick(Kind::Release, class, bytes);
        self.fake.release(class, bytes)
    }
}

/// Marks that the injected panics' messages are swallowed. The hook is set
/// once for the process and never restored: a guard that restored it on Drop
/// would call `set_hook` from a panicking thread, which aborts.
struct QuietHook;

fn quiet_injected() -> QuietHook {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let msg = info
                .payload()
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| info.payload().downcast_ref::<&str>().copied())
                .unwrap_or("");
            if !msg.contains("injected panic") {
                prev(info);
            }
        }));
    });
    QuietHook
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
        let every_call = self.chaos.every.load(Ordering::SeqCst) == 1;
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
            if every_call {
                // The one state in which the drain cannot run, and queued
                // items rightly keep their charges: a ledger that panics on
                // every call never lets a release succeed. A panic that
                // poisoned `active_m4` is not one: the recycle recovers it.
                // The books must still agree: what is queued is what the
                // ledger holds as Live, and nothing was released twice. The
                // runtime's Drop then returns them (checked below).
                let queued: u64 = rt
                    .pending_cold_recycle
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .iter()
                    .map(|(_, (b, _))| b.length() as u64)
                    .chain(
                        rt.pending_retirement
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .iter()
                            .map(|(_, b)| b.length() as u64),
                    )
                    .sum();
                assert_eq!(queued, fake.live(), "{ctx}: queued bytes vs ledger Live");
                assert_eq!(fake.unmatched(), 0, "{ctx}: an unmatched release");
            } else {
                assert!(!rt.deferred_pending(), "{ctx}: the drain left items queued");
            }
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
fn a_ledger_panic_inside_drop_still_returns_the_remaining_charges() {
    let _quiet = quiet_injected();
    let mut n = 1;
    loop {
        let mut rig = Rig::new();
        rig.seed();
        rig.apply(Op::DropAll);
        let Rig { rt, fake, chaos, held } = rig;
        drop(held);
        chaos.panic_on_call(n);
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(rt))).is_err();
        assert_eq!(
            (fake.live(), fake.cached()),
            (0, 0),
            "a panic at ledger call {n} inside Drop left charges on the ledger"
        );
        if !unwound {
            break;
        }
        n += 1;
    }
    // The free list and the drained queue hold at least four charges between them.
    assert!(n > 4, "Drop made only {} ledger calls", n - 1);
}

/// A runtime dropped while the thread is already unwinding must not resume
/// the ledger's panic (a second panic aborts the process). Every other charge
/// still comes back; the one whose release panicked stays held.
#[test]
fn a_runtime_dropped_during_an_unwind_returns_every_other_charge_and_does_not_abort() {
    let _quiet = quiet_injected();
    for n in 1..=6 {
        let mut rig = Rig::new();
        rig.seed();
        rig.apply(Op::DropAll);
        let Rig { rt, fake, chaos, held } = rig;
        drop(held);
        chaos.panic_on_call(n);
        let outer = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _runtime = rt;
            panic!("injected panic: the outer unwind");
        }));
        assert!(outer.is_err());
        assert_eq!(
            fake.live() + fake.cached(),
            chaos.panicked_bytes.load(Ordering::SeqCst),
            "a panic at ledger call {n} inside a Drop during an unwind"
        );
        assert_eq!(fake.unmatched(), 0, "call {n}: a charge was released twice");
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

/// A ledger panic inside a non-waiting commit's drain, with a command buffer
/// open and a dropped buffer queued behind it. The panic unwinds with
/// `active_m4` held, so the lock is left poisoned; the queued drop is
/// requeued by the drain's unwind guard, charge intact. The GPU has finished
/// everything submitted when this returns, so only the lock stands between
/// the queue and [`GpuRuntime::recycle_if_idle`].
fn poison_active_m4_in_a_commit_drain() -> Rig {
    let _quiet = quiet_injected();
    let rig = Rig::new();
    rig.rt.set_async_encode(true).unwrap();
    let buf = rig.rt.alloc_buffer(300_000).unwrap();
    rig.rt.with_binder(|_| Ok(())).unwrap();
    drop(buf);
    rig.chaos.panic_on_kind(Kind::Release, LedgerClass::Live, 1);
    let commit = |rig: &Rig| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rig.rt.commit(false))).is_err();
    let mut unwound = commit(&rig);
    if !unwound {
        // The GPU had not reached the commit's value when it checked, so the
        // drain found nothing to release. Wait, then commit again: the drain
        // now sees the queued drop as finished.
        rig.rt
            .wait_lifetime_event(rig.rt.metal4.event_value.load(Ordering::Acquire))
            .unwrap();
        rig.rt.with_binder(|_| Ok(())).unwrap();
        unwound = commit(&rig);
    }
    assert!(unwound, "the injected panic never reached a commit's drain");
    assert!(rig.rt.active_m4.is_poisoned(), "the unwind did not poison active_m4");
    rig.rt
        .wait_lifetime_event(rig.rt.metal4.event_value.load(Ordering::Acquire))
        .unwrap();
    rig.chaos.reset();
    assert!(rig.rt.deferred_pending(), "the panicked drain did not requeue the drop");
    rig.audit("after the panic");
    rig
}

/// A panic that poisoned `active_m4` must not park the deferred queues for
/// good: recycling at an idle moment gives the charge back, as it does when
/// the lock is clean.
#[test]
fn a_poisoned_batch_lock_does_not_stop_the_idle_recycle() {
    let rig = poison_active_m4_in_a_commit_drain();
    settle(&rig.rt);
    assert!(
        !rig.rt.deferred_pending(),
        "a poisoned active_m4 left the deferred queue undrained"
    );
    assert_eq!(rig.fake.live(), 0, "the drop's Live charge was never released");
    rig.audit("after the idle recycle");
    rig.finish(false, "poisoned active_m4");
}

/// A lock this thread holds is "a command buffer may be open" whether or not
/// it is poisoned: the recycle waits for the guard to go.
#[test]
fn a_poisoned_batch_lock_this_thread_holds_still_blocks_the_idle_recycle() {
    let rig = poison_active_m4_in_a_commit_drain();
    {
        let _batch = rig.rt.active_m4.lock().unwrap_or_else(PoisonError::into_inner);
        settle(&rig.rt);
        assert!(rig.rt.deferred_pending(), "recycled under a held batch lock");
    }
    settle(&rig.rt);
    assert!(!rig.rt.deferred_pending(), "not recycled once the lock was released");
    assert_eq!(rig.fake.live(), 0);
    rig.finish(false, "held poisoned active_m4");
}

// ---- adversarial pins for the poisoned `active_m4` recovery (ft-f68409c6) ----

/// Panic while holding the batch lock, as a panicking commit drain would,
/// without needing a ledger to do it.
fn poison_batch_lock(rt: &GpuRuntime) {
    let _quiet = quiet_injected();
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _held = rt.active_m4.lock().unwrap_or_else(PoisonError::into_inner);
        panic!("injected panic: poisoning active_m4 on purpose");
    }));
    assert!(unwound.is_err());
    assert!(rt.active_m4.is_poisoned());
}

/// A rig with a command buffer open (async encode, one empty scope) and one
/// dropped buffer queued behind it.
fn rig_with_open_batch_and_a_queued_drop() -> Rig {
    let rig = Rig::new();
    rig.rt.set_async_encode(true).unwrap();
    let buf = rig.rt.alloc_buffer(300_000).unwrap();
    rig.rt.with_binder(|_| Ok(())).unwrap();
    assert!(rig.rt.active_m4.lock().unwrap().as_ref().is_some_and(|b| b.cb_open));
    drop(buf);
    assert!(rig.rt.deferred_pending());
    rig
}

/// Attack 1. A lock poisoned while a command buffer is open: the recycle must
/// skip (the open buffer may bind the drop), and the queue is drainable once
/// `abort_open_batch` has ended the batch. Before the fix it stayed parked
/// even then.
#[test]
fn adv_poison_with_an_open_batch_skips_then_drains_after_the_abort() {
    let rig = rig_with_open_batch_and_a_queued_drop();
    poison_batch_lock(&rig.rt);
    settle(&rig.rt);
    assert!(rig.rt.deferred_pending(), "recycled under an open command buffer");
    assert_eq!(
        rig.rt.reclaim(u64::MAX).unwrap(),
        0,
        "reclaim released under an open buffer"
    );
    assert!(rig.fake.live() > 0);
    // An allocation does not drain it either.
    drop(rig.rt.alloc_buffer(2 << 20).unwrap());
    assert!(rig.rt.deferred_pending());

    rig.rt.abort_open_batch();
    assert!(rig.rt.active_m4.is_poisoned(), "the abort is not a clear_poison");
    settle(&rig.rt);
    assert!(!rig.rt.deferred_pending(), "still parked after the batch was aborted");
    rig.audit("after abort and recycle");
    rig.finish(false, "poisoned with an open batch, then aborted");
}

/// Attack 1, second half. With the batch open and the lock poisoned, nothing
/// but the abort drains the queue: a synchronize refuses (and poisons the
/// runtime), a reclaim refuses, a recycle skips. The runtime's Drop returns the
/// charge exactly once.
#[test]
fn adv_poison_with_an_open_batch_parks_the_queue_until_drop_and_drop_releases_once() {
    let rig = rig_with_open_batch_and_a_queued_drop();
    poison_batch_lock(&rig.rt);
    let err = rig.rt.synchronize().unwrap_err();
    assert!(
        err.contains("poison"),
        "synchronize did not report the poisoned lock: {err}"
    );
    assert!(rig.rt.is_poisoned());
    assert!(rig.rt.reclaim(u64::MAX).is_err());
    settle(&rig.rt);
    assert!(rig.rt.deferred_pending(), "the queue drained without the abort");
    rig.audit("parked");
    rig.finish(true, "poisoned with an open batch, dropped");
}

/// The refusal tells the caller the contract (ft-8f7aabe6): the batch lock was
/// poisoned by a panic, not recoverable, drop it, and dropping releases the parked queue
/// once. `synchronize` returns the lock refusal itself; `reclaim` then returns
/// the runtime latch, which carries that text as its first cause.
#[test]
fn a_mid_batch_poison_refusal_states_the_drop_contract() {
    let rig = rig_with_open_batch_and_a_queued_drop();
    poison_batch_lock(&rig.rt);
    let sync_err = rig.rt.synchronize().unwrap_err();
    let reclaim_err = rig.rt.reclaim(u64::MAX).unwrap_err();
    for (what, err) in [("synchronize", &sync_err), ("reclaim", &reclaim_err)] {
        for needle in ["poisoned", "batch lock", "panic", "drop it", "releases", "exactly once"] {
            assert!(err.contains(needle), "{what} refusal lacks {needle:?}: {err}");
        }
    }
    assert!(rig.rt.deferred_pending(), "the queue drained without the abort");
    rig.finish(true, "mid-batch poison text, dropped");
}

// ---- adversarial pins for the drop-it contract text (ft-8f7aabe6, w6-adv-poison) ----

type Entry = (
    &'static str,
    Box<dyn Fn(&Rig, &crate::tensor::GpuBuffer) -> Result<(), String>>,
);

/// Every public way into a runtime whose `active_m4` was poisoned with a
/// command buffer open, each tried on its own fresh rig (the first refusal
/// latches the runtime, so a shared rig would answer later entries from the
/// latch). `reclaim` is not here: it does not take the lock (see the
/// `reclaim_is_ok_zero_...` test below).
fn poisoned_batch_entries() -> Vec<Entry> {
    vec![
        ("synchronize", Box::new(|r, _| r.rt.synchronize())),
        ("commit(false)", Box::new(|r, _| r.rt.commit(false))),
        ("commit(true)", Box::new(|r, _| r.rt.commit(true))),
        ("with_binder", Box::new(|r, _| r.rt.with_binder(|_| Ok(())))),
        (
            "with_binder_barriers",
            Box::new(|r, _| r.rt.with_binder_barriers(Some(true), |_| Ok(()))),
        ),
        ("set_async_encode(false)", Box::new(|r, _| r.rt.set_async_encode(false))),
        (
            "upload_u32",
            Box::new(|r, b| r.rt.upload_u32(b, &vec![1u32; b.nbytes() / 4])),
        ),
        ("GpuBuffer::try_read_f32", Box::new(|_, b| b.try_read_f32().map(|_| ()))),
        ("GpuBuffer::try_zero", Box::new(|_, b| b.try_zero())),
        ("bump_reset", Box::new(|r, _| r.rt.bump_reset())),
    ]
}

/// Attack (a). After a mid-batch poison, every entry that reaches the batch
/// lock answers with the contract text (directly, or as the latch's cause).
/// The table prints what each returns.
#[test]
fn adv_every_public_entry_refuses_a_mid_batch_poison_with_the_contract_text() {
    let mut rows = Vec::new();
    for (name, call) in poisoned_batch_entries() {
        let rig = rig_with_open_batch_and_a_queued_drop();
        let buf = rig.rt.alloc_buffer(4096).unwrap();
        poison_batch_lock(&rig.rt);
        let got = call(&rig, &buf);
        let row = match &got {
            Ok(()) => format!("{name}: Ok"),
            Err(e) => format!("{name}: Err({e})"),
        };
        println!("{row}");
        rows.push(row);
        let err = got.unwrap_err();
        for needle in ["poisoned", "batch lock", "drop it", "exactly once"] {
            assert!(err.contains(needle), "{name} refusal lacks {needle:?}: {err}");
        }
        assert!(rig.rt.deferred_pending(), "{name}: the queue drained without the abort");
        drop(buf);
        rig.finish(true, "entry table");
    }
    assert_eq!(rows.len(), 10);
}

/// The accessors that do not take the lock keep answering after the poison.
#[test]
fn adv_the_lock_free_accessors_still_answer_after_a_mid_batch_poison() {
    let rig = rig_with_open_batch_and_a_queued_drop();
    poison_batch_lock(&rig.rt);
    assert!(!rig.rt.is_poisoned(), "a bare lock poison latches nothing");
    let _ = rig.rt.pool_resident();
    let _ = rig.rt.pending_commit_value();
    rig.rt.set_async_encode(true).unwrap();
    drop(rig.rt.alloc_buffer(4096).unwrap());
    assert!(rig.rt.deferred_pending());
    rig.finish(true, "lock-free accessors");
}

/// Attack (a), pinned as the docs on `GpuRuntime` and `reclaim` state it: on a
/// lock poisoned by a panic that no refusal has latched yet (a panic unwinds
/// through `commit`, skipping its `poison_on_failure`), `reclaim` never touches
/// `active_m4` and returns Ok(0) with the queue still parked; once a refusal
/// latches the runtime it errors with the contract text as the cause.
#[test]
fn reclaim_is_ok_zero_on_a_fresh_mid_batch_poison_then_errors_once_a_refusal_latches() {
    let rig = rig_with_open_batch_and_a_queued_drop();
    poison_batch_lock(&rig.rt);
    // Phase 1, as the docs state: no refusal has latched, reclaim does not take
    // the lock, so it releases nothing parked.
    assert!(!rig.rt.is_poisoned());
    assert_eq!(rig.rt.reclaim(u64::MAX), Ok(0), "reclaim before any refusal");
    assert!(rig.rt.deferred_pending(), "reclaim released under an open buffer");
    // Phase 2: the first refusal latches, and reclaim answers from the latch with
    // the contract text as its cause.
    assert!(rig.rt.synchronize().is_err());
    assert!(rig.rt.is_poisoned());
    let got = rig.rt.reclaim(u64::MAX);
    assert!(
        got.as_ref().is_err_and(|e| e.contains("drop it")),
        "reclaim after the latch answered {got:?}"
    );
    rig.finish(true, "reclaim fresh poison");
}

/// Attack (b). A poison with no command buffer open (a panic in a waiting
/// commit's drain with no batch, as `poison_active_m4_in_a_commit_drain`
/// stages): the refusals are the same and the runtime still cannot encode or
/// commit, but the parked queue is NOT parked: the idle recycle drains it, and
/// Drop returns the charges once.
#[test]
fn adv_a_poison_with_no_batch_open_refuses_the_same_and_still_recycles() {
    let rig = Rig::new();
    drop(rig.rt.alloc_buffer(300_000).unwrap());
    assert!(rig.rt.active_m4.lock().unwrap().is_none());
    poison_batch_lock(&rig.rt);
    let err = rig.rt.synchronize().unwrap_err();
    println!("closed-case synchronize: {err}");
    assert!(err.contains("poisoned") && err.contains("drop it"), "{err}");
    assert!(rig.rt.with_binder(|_| Ok(())).is_err());
    settle(&rig.rt);
    assert!(!rig.rt.deferred_pending(), "closed-case poison parked the queue");
    rig.finish(true, "closed poison");
}

/// Attack (b). With no batch at all the refusal must not name a mid-batch state
/// or an open command buffer that never existed (the text covers both cases).
#[test]
fn the_refusal_does_not_claim_mid_batch_when_no_batch_exists() {
    let rig = Rig::new();
    assert!(rig.rt.active_m4.lock().unwrap().is_none());
    poison_batch_lock(&rig.rt);
    let err = rig.rt.synchronize().unwrap_err();
    assert!(!err.contains("mid-batch"), "no batch was open, yet: {err}");
    assert!(!err.contains("command buffer"), "no batch was open, yet: {err}");
    assert!(err.contains("poisoned") && err.contains("drop it"), "{err}");
    rig.finish(true, "closed poison text");
}

/// Attack (b). Poison, refuse, clear, poison again, refuse, abort twice, then
/// Drop in both orders: the charges come back once each, none twice.
#[test]
fn adv_a_second_poison_then_drop_releases_exactly_once() {
    for runtime_first in [true, false] {
        let rig = rig_with_open_batch_and_a_queued_drop();
        poison_batch_lock(&rig.rt);
        assert!(rig.rt.with_binder(|_| Ok(())).is_err());
        rig.rt.active_m4.clear_poison();
        poison_batch_lock(&rig.rt);
        assert!(rig.rt.synchronize().is_err());
        rig.rt.abort_open_batch();
        rig.rt.abort_open_batch();
        assert!(rig.rt.deferred_pending());
        let Rig { held, rt, fake, chaos } = rig;
        if runtime_first {
            drop(rt);
            drop(held);
        } else {
            drop(held);
            drop(rt);
        }
        drop(chaos);
        assert_eq!(
            (fake.live(), fake.cached(), fake.unmatched()),
            (0, 0, 0),
            "runtime_first={runtime_first}"
        );
    }
}

/// Attack 2. The recovery must still wait for the lifetime event: a commit
/// submitted and not finished keeps a dropped buffer queued, whether the lock
/// is poisoned before or after the drop, and a partial completion is not
/// enough.
#[test]
fn adv_poisoned_recycle_never_releases_under_an_unfinished_commit() {
    let rig = Rig::new();
    let buf = rig.rt.alloc_buffer(300_000).unwrap();
    let first = rig.rt.stage_submitted_commit_for_test();
    let second = rig.rt.stage_submitted_commit_for_test();
    drop(buf);
    poison_batch_lock(&rig.rt);
    for _ in 0..3 {
        settle(&rig.rt);
        assert!(rig.rt.deferred_pending(), "released while nothing had completed");
    }
    rig.rt.metal4.lifetime_event.setSignaledValue(first);
    settle(&rig.rt);
    assert!(
        rig.rt.deferred_pending(),
        "released with the second commit still running"
    );
    assert_eq!(rig.rt.reclaim(u64::MAX).unwrap(), 0);
    rig.rt.metal4.lifetime_event.setSignaledValue(second);
    settle(&rig.rt);
    assert!(!rig.rt.deferred_pending());
    rig.finish(false, "poisoned recycle under an unfinished commit");
}

/// Attack 2, with a dishonest event: a signal above anything submitted must
/// poison and refuse, not count as completion, even on the recovered path.
#[test]
fn adv_poisoned_recycle_still_refuses_a_dishonest_lifetime_event() {
    let rig = Rig::new();
    drop(rig.rt.alloc_buffer(300_000).unwrap());
    poison_batch_lock(&rig.rt);
    let submitted = rig.rt.metal4.event_value.load(Ordering::Acquire);
    rig.rt.metal4.lifetime_event.setSignaledValue(submitted + 1000);
    assert!(
        rig.rt.recycle_if_idle().is_err(),
        "a dishonest event counted as completion"
    );
    assert!(rig.rt.deferred_pending());
    assert_eq!(rig.fake.live(), bucket(300_000, BufferKind::Cold));
    // Restore an honest value so the books can close.
    rig.rt.metal4.lifetime_event.setSignaledValue(submitted);
    rig.finish(true, "dishonest event after poison");
}

/// Attack 3. Poison, recycle with buffers still held, then both Drop orders:
/// every charge comes back exactly once and nothing is unmatched.
#[test]
fn adv_poison_recycle_then_drop_balances_the_books_in_both_orders() {
    for runtime_first in [false, true] {
        let mut rig = Rig::new();
        rig.seed();
        poison_batch_lock(&rig.rt);
        rig.apply(Op::DropAll);
        settle(&rig.rt);
        assert!(!rig.rt.deferred_pending());
        rig.audit("after the poisoned recycle");
        // A buffer taken after the recovery is live across the Drop.
        rig.apply(Op::Hit);
        rig.apply(Op::Miss);
        rig.audit("with buffers live again");
        rig.finish(runtime_first, "poison, recycle, drop");
    }
}

/// Attack 3. A queue the recycle never ran on is returned by Drop, once.
#[test]
fn adv_poison_then_drop_without_a_recycle_releases_each_queued_charge_once() {
    let mut rig = Rig::new();
    rig.seed();
    rig.apply(Op::DropAll);
    assert!(rig.rt.deferred_pending());
    poison_batch_lock(&rig.rt);
    let Rig { held, rt, fake, chaos } = rig;
    drop(held);
    drop(rt);
    drop(chaos);
    assert_eq!((fake.live(), fake.cached(), fake.unmatched()), (0, 0, 0));
}

/// Attack 4. Another thread cannot hold the lock while this one recycles:
/// `GpuRuntime` is `!Send + !Sync` (E0277 on sharing it into `thread::scope`).
/// The same `WouldBlock` is reached on one thread by holding the guard, clean
/// or poisoned, across the recycle; poison then goes with the guard's drop.
#[test]
fn adv_a_batch_lock_held_across_the_recycle_skips_even_when_poisoned() {
    for poisoned in [false, true] {
        let rig = Rig::new();
        drop(rig.rt.alloc_buffer(300_000).unwrap());
        if poisoned {
            poison_batch_lock(&rig.rt);
        }
        {
            let _held = rig.rt.active_m4.lock().unwrap_or_else(PoisonError::into_inner);
            settle(&rig.rt);
            assert!(
                rig.rt.deferred_pending(),
                "poisoned={poisoned}: recycled under a held guard"
            );
            assert_eq!(rig.rt.reclaim(u64::MAX).map(|_| ()), Ok(()));
        }
        settle(&rig.rt);
        assert!(
            !rig.rt.deferred_pending(),
            "poisoned={poisoned}: not recycled once the guard was gone"
        );
        rig.finish(false, "lock held across the recycle");
    }
}

/// Attack 5. Poison, recycle, poison again (cleared in between, and not): each
/// recycle still drains.
#[test]
fn adv_repeated_poison_still_recycles_each_time() {
    for clear_between in [false, true] {
        let rig = Rig::new();
        for round in 0..5 {
            drop(rig.rt.alloc_buffer(300_000).unwrap());
            if clear_between {
                rig.rt.active_m4.clear_poison();
            }
            poison_batch_lock(&rig.rt);
            settle(&rig.rt);
            assert!(
                !rig.rt.deferred_pending(),
                "clear_between={clear_between} round {round}: parked"
            );
            rig.audit("repeated poison");
        }
        rig.finish(false, "repeated poison");
    }
}

/// Attack 6. After the lock is poisoned (and after a recycle recovered it), the
/// callers that encode or commit still refuse with the poison error, and the
/// closure never runs.
#[test]
fn adv_the_refusing_paths_still_refuse_a_poisoned_batch_lock() {
    for recycled_first in [false, true] {
        for async_encode in [false, true] {
            let ran = AtomicBool::new(false);
            let rig = Rig::new();
            rig.rt.set_async_encode(async_encode).unwrap();
            drop(rig.rt.alloc_buffer(300_000).unwrap());
            poison_batch_lock(&rig.rt);
            if recycled_first {
                settle(&rig.rt);
                assert!(!rig.rt.deferred_pending());
            }
            let ctx = format!("recycled_first={recycled_first} async={async_encode}");
            // The read-only reader recovers.
            let _ = rig.rt.pending_commit_value();
            let err = rig
                .rt
                .with_binder(|_| {
                    ran.store(true, Ordering::SeqCst);
                    Ok(())
                })
                .unwrap_err();
            assert!(err.contains("poison"), "{ctx}: with_binder error was {err}");
            assert!(
                !ran.load(Ordering::SeqCst),
                "{ctx}: the closure ran on a poisoned batch"
            );
            assert!(rig.rt.active_m4.is_poisoned(), "{ctx}: a refusal cleared the poison");
            rig.finish(true, "refusing paths");
        }
    }
    for wait in [false, true] {
        let rig = Rig::new();
        poison_batch_lock(&rig.rt);
        settle(&rig.rt);
        let err = rig.rt.commit(wait).unwrap_err();
        assert!(err.contains("poison"), "commit({wait}) error was {err}");
        assert!(rig.rt.synchronize().is_err());
        rig.finish(true, "refusing commit");
    }
}

/// `cb_open == false` must imply that `event_value` covers every command
/// buffer this batch submitted: `recycle_if_idle` recovers a poisoned
/// `active_m4`, reads the flag, and then releases everything deferred against
/// the event value. Clearing the flag before the store leaves a window in
/// which a panic poisons the lock with the command buffer submitted and the
/// event value stale, and the recycle frees buffers the GPU is still reading.
/// A source-order pin: no test can make the panic land in that window without
/// a production knob.
#[test]
fn commit_m4_clears_cb_open_only_after_event_value_covers_the_submit() {
    let src = include_str!("runtime.rs");
    let start = src.find("fn commit_m4(").expect("commit_m4 not found");
    let body = &src[start..];
    let end = body[1..]
        .find("\n    fn ")
        .or_else(|| body[1..].find("\n    pub fn "))
        .expect("end of commit_m4 not found");
    let body = &body[..=end];
    let store = body
        .find("event_value.store(next")
        .expect("commit_m4 no longer stores the event value");
    let clears: Vec<usize> = body.match_indices("cb_open = false").map(|(i, _)| i).collect();
    assert!(!clears.is_empty(), "commit_m4 no longer clears cb_open");
    for at in clears {
        assert!(
            store < at,
            "commit_m4 clears cb_open (byte {at}) before it stores event_value (byte {store})"
        );
    }
}

// ---- a commit that fails before its submit (ft-2ea050a0) ----

/// A rig with a command buffer open and nothing encoded, and the end counter
/// at zero.
fn rig_with_an_open_empty_batch() -> Rig {
    let rig = Rig::new();
    rig.rt.set_async_encode(true).unwrap();
    rig.rt.with_binder(|_| Ok(())).unwrap();
    assert!(rig.rt.active_m4.lock().unwrap().as_ref().is_some_and(|b| b.cb_open));
    END_CALLS.with(|c| c.set(0));
    rig
}

fn end_calls() -> u64 {
    END_CALLS.with(Cell::get)
}

fn batch_is_open(rt: &GpuRuntime) -> bool {
    let guard = rt.active_m4.lock().unwrap_or_else(PoisonError::into_inner);
    guard.as_ref().is_some_and(|b| b.cb_open)
}

/// A commit that returns `Err` before anything was submitted must leave the
/// command buffer open: `cb_open` says "recording, not submitted", and the
/// drop-time abort ends it once. Before the fix the buffer was ended first and
/// the abort ended it again.
#[test]
fn a_commit_that_errs_before_the_submit_leaves_the_buffer_open_and_ends_it_once() {
    let rig = rig_with_an_open_empty_batch();
    let submitted = rig.rt.metal4.event_value.load(Ordering::Acquire);
    COMMIT_FAULT.with(|f| f.set(Some(CommitFault::Err)));
    let err = rig.rt.commit(false).unwrap_err();
    assert!(err.contains("injected commit fault"), "error was {err}");
    let (open, ended) = (batch_is_open(&rig.rt), end_calls());
    eprintln!("after the failed commit: cb_open={open} end_calls={ended}");
    assert!(open, "a failed commit cleared cb_open");
    assert_eq!(ended, 0, "the buffer was ended before the commit could fail");
    assert_eq!(
        rig.rt.metal4.event_value.load(Ordering::Acquire),
        submitted,
        "a commit that never submitted moved event_value"
    );
    drop(rig);
    eprintln!("after the drop: end_calls={}", end_calls());
    assert_eq!(end_calls(), 1, "the buffer must be ended exactly once, by the abort");
}

/// The same for a panic: the unwind poisons the batch lock with `cb_open`
/// true, which is the truth, and the drop-time abort ends the buffer once.
#[test]
fn a_commit_that_panics_before_the_submit_leaves_the_buffer_open_and_ends_it_once() {
    let _quiet = quiet_injected();
    let rig = rig_with_an_open_empty_batch();
    let submitted = rig.rt.metal4.event_value.load(Ordering::Acquire);
    COMMIT_FAULT.with(|f| f.set(Some(CommitFault::Panic)));
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rig.rt.commit(false))).is_err();
    assert!(unwound, "the injected panic never fired");
    assert!(rig.rt.active_m4.is_poisoned());
    let (open, ended) = (batch_is_open(&rig.rt), end_calls());
    eprintln!("after the panicked commit: cb_open={open} end_calls={ended}");
    assert!(open, "a panicked commit cleared cb_open");
    assert_eq!(ended, 0, "the buffer was ended before the commit could panic");
    assert_eq!(rig.rt.metal4.event_value.load(Ordering::Acquire), submitted);
    drop(rig);
    eprintln!("after the drop: end_calls={}", end_calls());
    assert_eq!(end_calls(), 1, "the buffer must be ended exactly once, by the abort");
}

/// Code lines of `src` from `header` to the first line that closes the
/// function (a lone four-space `}`), comments removed.
fn code_of_fn(src: &str, header: &str) -> String {
    let start = src.find(header).unwrap_or_else(|| panic!("{header} not found"));
    let rest = &src[start..];
    let end = rest
        .find("\n    }\n")
        .unwrap_or_else(|| panic!("end of {header} not found"));
    rest[..end]
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

const FAILURE_POINTS: [&str; 8] = [
    "?",
    "unwrap",
    "expect(",
    "panic!",
    "assert",
    "unreachable!",
    "todo!",
    "ok_or",
];

/// A source-order pin (no test can fire a panic inside the window without a
/// production knob): between ending the command buffer and clearing `cb_open`,
/// `commit_m4` has no `?` and no panic site, and `submit_prepared`, which it
/// calls there, has none either. All fallible work, and the clock check, runs in
/// `prepare_commit` before the end, while a failure still leaves `cb_open` true.
#[test]
fn commit_m4_has_no_failure_point_between_ending_the_buffer_and_clearing_cb_open() {
    let src = include_str!("runtime.rs");
    let body = code_of_fn(src, "fn commit_m4(");
    let prepare = body
        .find("self.prepare_commit()?")
        .expect("commit_m4 no longer prepares the commit");
    let guard = body
        .find("debug_assert!(")
        .expect("commit_m4 lost the clock check before the end");
    let end = body
        .find("self.end_command_buffer();")
        .expect("commit_m4 no longer ends the command buffer through end_command_buffer");
    let clear_rel = body[end..]
        .find("cb_open = false")
        .expect("no cb_open clear after the end");
    let window = &body[end..end + clear_rel];
    assert!(
        prepare < end && guard < end,
        "the fallible prepare step and the clock check must run before the end"
    );
    assert!(
        window.contains("self.submit_prepared(") && window.contains("event_value.store(next"),
        "the window lost the submit or the event_value store"
    );
    for point in FAILURE_POINTS {
        assert!(
            !window.contains(point),
            "commit_m4 has a failure point ({point}) between end_command_buffer and the cb_open clear:\n{window}"
        );
    }
    let submit = code_of_fn(src, "fn submit_prepared(");
    for point in FAILURE_POINTS {
        assert!(
            !submit.contains(point),
            "submit_prepared has a failure point ({point}):\n{submit}"
        );
    }
}
