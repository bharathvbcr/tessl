//! State that several [`GpuRuntime`]s of one process can share: the poison
//! latch and the commit clock, behind one `Arc<Shared>`.
//!
//! `Shared` is `Send + Sync` with no `unsafe`: every field is an atomic or a
//! `Mutex` over plain data, and the compile-time assertion at the bottom of
//! this file keeps it that way. What it does **not** hold is anything that
//! objc2 leaves `!Send`:
//!
//! - the buffer pool and the deferred-release queues hold
//!   `Retained<ProtocolObject<dyn MTLBuffer>>`, and `MTLBuffer` carries no
//!   `Send`/`Sync` marker in objc2-metal (`MTLEvent` does, `MTLBuffer` does
//!   not). They stay in the runtime until the owner decides who may assert
//!   thread-safety for a buffer wrapper (decision D6 of the unification plan).
//! - the queue's lifetime event stays with its queue. Two queues signalling
//!   one event with interleaved values can finish out of order and leave the
//!   event's value lower than a value already waited for, so each runtime keeps
//!   its own event and the runtimes share only the *numbers* they sign it with.
//!
//! [`GpuRuntime::timeline`] is the read-and-wait view over one runtime's
//! private lifetime event.
//!
//! # Per-buffer stamps are not here yet: ICB replay
//!
//! A per-bind `last_use` stamp set where a buffer is bound (`Binder::bind_owned`)
//! would be **wrong on a replayed command buffer**, which never goes through it.
//! [`crate::decode_icb::DecodeIcb::execute`] rebinds with raw addresses
//! (`StickyArgTable::bind_addr` into `Binder::bind_addr`) or, by default, adopts
//! a prebuilt per-command argument table whose buffer binds were frozen at
//! `from_commands` and are skipped at execute (`bind_prebuilt` counts them as
//! `prebuilt_elided`). Neither calls `bind_owned`, so a buffer a tape reads or
//! writes would keep the stamp of its capture while a replay is in flight, and
//! a host write that waits for `completed >= last_use` could start under it.
//!
//! What re-stamping needs, none of which exists yet:
//! 1. each tape records, at `from_commands`, its distinct buffers with a role
//!    (read-only or read-write); `DecodeIcbBind::Buf` carries no role today and
//!    the capture notes only `(index, buf, offset)`, so the role has to come
//!    from the role-aware bind (`bind_shared_ro/rw` of the plan) or default to
//!    read-write, which is safe and only costs a stricter host wait;
//! 2. every non-nop `execute` stamps those buffers with the commit value that
//!    will carry the replay, one pass over the distinct buffers rather than the
//!    binds;
//! 3. that value is read inside the `with_binder_barriers` closure, while the
//!    command buffer that carries the replay is open:
//!    `GpuRuntime::pending_commit_value` names `event_value + 1` only while one
//!    is, and a mid-commit after the closure (`encode_into_batch_m4_inner`,
//!    `hit_mid`) would make a later read name the next command buffer. The
//!    closure holds `const_cursor`, not `active_m4`, so the read does not
//!    deadlock (read, `runtime.rs` `encode_into_batch_m4_inner`);
//! 4. a replay suppressed by binder-encode-nop stamps nothing, because it
//!    encoded nothing.
//!
//! `DecodeIcb::execute` also takes any `&GpuRuntime` and never checks that it
//! is the runtime the tape was built on (`from_commands` checks, `execute`
//! does not), so a stamp taken from the wrong runtime would be a wrong clock.
//! Whoever lands the stamps adds that check first.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use crate::runtime::GpuRuntime;

/// The poison latch: set once an encode, a submit, a GPU fault, or the third
/// consecutive wait-budget timeout fails, after which the runtime refuses
/// every call. One or two budget timeouts do not set it. It keeps the first
/// cause, because only the call that poisoned the runtime returns it; every
/// later refusal would otherwise say only "poisoned", far from the fault.
///
/// Metal's feedback thread sets it too, where a panic cannot be recovered, so
/// the lock takes a poisoned guard as it is.
pub(crate) struct RuntimePoison {
    set: AtomicBool,
    cause: Mutex<Option<String>>,
}

impl RuntimePoison {
    pub(crate) fn new() -> Self {
        Self {
            set: AtomicBool::new(false),
            cause: Mutex::new(None),
        }
    }

    /// Poison the runtime. A later cause does not replace the first.
    pub(crate) fn set(&self, cause: impl Into<String>) {
        {
            let mut first = self.cause.lock().unwrap_or_else(PoisonError::into_inner);
            if first.is_none() {
                *first = Some(cause.into());
            }
        }
        self.set.store(true, Ordering::Release);
    }

    pub(crate) fn is_set(&self) -> bool {
        self.set.load(Ordering::Acquire)
    }

    /// The refusal a poisoned runtime returns, naming the first cause.
    pub(crate) fn error(&self) -> String {
        match &*self.cause.lock().unwrap_or_else(PoisonError::into_inner) {
            Some(cause) => format!("runtime is poisoned after encode/submit failure ({cause}); recreate it"),
            None => "runtime is poisoned after encode/submit failure; recreate it".into(),
        }
    }
}

/// What the runtimes of one process share. See the module docs for what it
/// leaves out and why.
///
/// A runtime made with [`GpuRuntime::new`] gets a `Shared` of its own, so a
/// lone runtime behaves exactly as it did before this type existed. Runtimes
/// made with [`GpuRuntime::with_shared`] from one `Arc<Shared>` share:
///
/// - **poison.** A fault or a spent wait budget in one refuses every call on
///   all of them, and the first cause is what each refusal reports.
/// - **the commit clock.** Every commit of every runtime draws its timeline
///   value from one counter, so the values are unique and ordered by when the
///   commits were made across all the runtimes. A runtime's own sequence is
///   still strictly increasing, but no longer contiguous: after a commit of
///   another runtime, "the next commit" of this one is some value above
///   `last_signaled_value() + 1`, not that value exactly.
pub struct Shared {
    pub(crate) poison: Arc<RuntimePoison>,
    issued: AtomicU64,
}

impl Shared {
    /// A fresh latch (not poisoned) and a clock at zero.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            poison: Arc::new(RuntimePoison::new()),
            issued: AtomicU64::new(0),
        })
    }

    /// Whether any runtime under this `Shared` has poisoned it.
    pub fn is_poisoned(&self) -> bool {
        self.poison.is_set()
    }

    /// The first poison cause, or `None` while the latch is clear.
    pub fn poison_error(&self) -> Option<String> {
        self.poison.is_set().then(|| self.poison.error())
    }

    /// The highest timeline value any runtime under this `Shared` has drawn
    /// for a commit (`0` before the first). Never decreases.
    pub fn issued(&self) -> u64 {
        self.issued.load(Ordering::Acquire)
    }

    /// Draw the next commit's timeline value: above every value drawn before.
    pub(crate) fn issue(&self) -> u64 {
        self.issued.fetch_add(1, Ordering::AcqRel) + 1
    }
}

/// The timeline view of one runtime's private lifetime event: what it has
/// submitted, what tessl has seen complete, and a bounded wait.
///
/// The view carries values, not the event. A caller cannot signal it, and a
/// foreign signal on the public handoff event cannot raise `completed`
/// (see [`GpuRuntime::last_completed_value`]).
#[derive(Clone, Copy)]
pub struct Timeline<'a> {
    rt: &'a GpuRuntime,
}

/// Start of the error [`Timeline::wait`] returns for a value no commit of this
/// runtime has been made for. Test for it with [`Timeline::is_not_submitted`].
const NOT_SUBMITTED_PREFIX: &str = "timeline value not submitted";

impl<'a> Timeline<'a> {
    pub(crate) fn new(rt: &'a GpuRuntime) -> Self {
        Self { rt }
    }

    /// The last value this runtime has submitted a signal for (`0` before
    /// its first commit).
    pub fn submitted(&self) -> u64 {
        self.rt.last_signaled_value()
    }

    /// The highest submitted value tessl has seen complete. Never above
    /// [`Self::submitted`].
    pub fn completed(&self) -> u64 {
        self.rt.last_completed_value()
    }

    /// Block until `value` has completed, for at most the runtime's
    /// [`GpuRuntime::event_wait_ms`] budget, through the same wait every
    /// lifetime decision uses (so the third consecutive timeout poisons).
    ///
    /// A poisoned runtime refuses first. A `value` above [`Self::submitted`]
    /// is refused with an error [`Self::is_not_submitted`] recognises, never
    /// waited on: no commit exists that would signal it, so the wait could
    /// only spend the budget. Flush through the runtime (`commit(false)`) and
    /// ask again.
    pub fn wait(&self, value: u64) -> Result<(), String> {
        self.rt.ensure_not_poisoned()?;
        if value == 0 {
            return Ok(());
        }
        let submitted = self.submitted();
        if value > submitted {
            return Err(format!(
                "{NOT_SUBMITTED_PREFIX}: asked for {value}, this runtime has submitted {submitted}"
            ));
        }
        self.rt.wait_lifetime_event(value)
    }

    /// Whether `error` is the refusal of [`Self::wait`] for an unsubmitted value.
    pub fn is_not_submitted(error: &str) -> bool {
        error.starts_with(NOT_SUBMITTED_PREFIX)
    }
}

/// `Shared` is shared across threads without `unsafe`; adding a `!Send` field
/// (a `Retained<MTLBuffer>`, a `Cell`) stops this file compiling.
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    send_sync::<Shared>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_poison_cause_is_kept_for_every_holder() {
        let shared = Shared::new();
        let a = Arc::clone(&shared.poison);
        let b = Arc::clone(&shared.poison);
        assert!(!shared.is_poisoned());
        assert_eq!(shared.poison_error(), None);
        a.set("first");
        b.set("second");
        assert!(shared.is_poisoned());
        let error = shared.poison_error().expect("poisoned");
        assert!(error.contains("first") && !error.contains("second"), "{error}");
        assert_eq!(error, b.error());
    }

    #[test]
    fn issued_values_are_unique_and_increase_across_threads() {
        let shared = Shared::new();
        let per_thread = 1_000;
        let drawn: Vec<Vec<u64>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let shared = Arc::clone(&shared);
                    s.spawn(move || (0..per_thread).map(|_| shared.issue()).collect::<Vec<u64>>())
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("thread")).collect()
        });
        for run in &drawn {
            assert!(
                run.windows(2).all(|w| w[0] < w[1]),
                "a thread saw a non-increasing draw"
            );
        }
        let mut all: Vec<u64> = drawn.into_iter().flatten().collect();
        all.sort_unstable();
        let expected: Vec<u64> = (1..=4 * per_thread as u64).collect();
        assert_eq!(all, expected, "draws are unique and leave no gap");
        assert_eq!(shared.issued(), 4 * per_thread as u64);
    }

    #[test]
    fn shared_crosses_threads() {
        fn needs_send_sync<T: Send + Sync + 'static>(_: &T) {}
        let shared = Shared::new();
        needs_send_sync(&shared);
        let moved = Arc::clone(&shared);
        std::thread::spawn(move || moved.poison.set("from another thread"))
            .join()
            .expect("thread");
        assert!(shared.is_poisoned());
    }

    #[test]
    fn the_not_submitted_refusal_is_recognised_and_other_errors_are_not() {
        let refusal = format!("{NOT_SUBMITTED_PREFIX}: asked for 5, this runtime has submitted 3");
        assert!(Timeline::is_not_submitted(&refusal));
        assert!(!Timeline::is_not_submitted(
            "runtime is poisoned after encode/submit failure; recreate it"
        ));
        assert!(!GpuRuntime::is_event_wait_timeout(&refusal));
    }
}
