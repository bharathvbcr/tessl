//! The borrow-free, thread-safe view of one runtime's private lifetime event.
//!
//! [`GpuRuntime`] is `!Send` and `!Sync` (Metal encoders are thread-affine), so
//! a CPU worker thread cannot ask it whether the GPU has finished with a
//! buffer. [`HostFence`] is what such a thread can hold instead: a clone of the
//! lifetime event (`MTLEvent` is `Send + Sync` in objc2-metal) and the `Arc`
//! atomics the runtime's own waits already use, so the budget, the timeout
//! streak, the submitted counter and the poison latch are *the same objects*
//! the runtime reads, not copies.
//!
//! It is a child module of `runtime` on purpose. The runtime's wait
//! ([`GpuRuntime::wait_lifetime_event`]) and its honesty check
//! ([`GpuRuntime::checked_lifetime_value`]) delegate to the methods here, so
//! there is one implementation of "wait for the lifetime event", not one for
//! the encoding thread and a second for host threads.
//!
//! Waits go through the private lifetime event only. The public handoff event
//! can be signalled by foreign code, so a value read from it proves nothing
//! about tessl's own work.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLSharedEvent;

use super::{
    event_wait_timeout_error, lifetime_event_violation, GpuRuntime, RuntimePoison, CONSECUTIVE_EVENT_WAIT_TIMEOUTS,
};
use crate::shared_buf::HostAccessError;

/// See the module docs. Cloning clones handles, not state.
#[derive(Clone)]
pub(crate) struct HostFence {
    lifetime_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// The runtime's `event_value`: the last commit value submitted.
    submitted: Arc<AtomicU64>,
    /// The runtime's wait budget in milliseconds.
    wait_ms: Arc<AtomicU64>,
    /// The runtime's consecutive-timeout streak.
    timeouts: Arc<AtomicU32>,
    poison: Arc<RuntimePoison>,
}

impl HostFence {
    pub(super) fn new(rt_parts: FenceParts<'_>) -> Self {
        Self {
            lifetime_event: rt_parts.lifetime_event.clone(),
            submitted: Arc::clone(rt_parts.submitted),
            wait_ms: Arc::clone(rt_parts.wait_ms),
            timeouts: Arc::clone(rt_parts.timeouts),
            poison: Arc::clone(rt_parts.poison),
        }
    }

    /// The poison error if the runtime is poisoned, else `Ok`.
    pub(crate) fn ensure_not_poisoned(&self) -> Result<(), String> {
        if self.poison.is_set() {
            return Err(self.poison.error());
        }
        Ok(())
    }

    /// The last commit value this runtime has submitted (`0` before the first).
    pub(crate) fn submitted(&self) -> u64 {
        self.submitted.load(Ordering::Acquire)
    }

    /// The lifetime event's value, after checking it against what the runtime
    /// submitted. A value above that was set by something other than the queue
    /// and cannot be evidence of completion, so the runtime poisons and
    /// refuses the lifetime decision that was about to rest on it.
    pub(crate) fn checked_value(&self) -> Result<u64, String> {
        // Signaled first, then submitted: `event_value` is bumped before the
        // queue signals, so an honest event never reads above the second load.
        let signaled = self.lifetime_event.signaledValue();
        let submitted = self.submitted.load(Ordering::Acquire);
        match lifetime_event_violation(signaled, submitted) {
            Some(error) => {
                self.poison.set(error.as_str());
                Err(error)
            }
            None => Ok(signaled),
        }
    }

    /// Block until the lifetime event reaches `value` (a value the runtime
    /// submitted), for at most the runtime's wait budget. The one wait every
    /// lifetime decision goes through.
    ///
    /// A budget timeout leaves the runtime usable. The third consecutive one
    /// poisons it (DeviceLost) and later calls return that poison without
    /// waiting out another budget. A wait that reaches `value` clears the streak.
    pub(super) fn wait(&self, value: u64) -> Result<(), String> {
        if self.poison.is_set() {
            return Err(self.poison.error());
        }
        let budget = self.wait_ms.load(Ordering::Acquire);
        if !self.lifetime_event.waitUntilSignaledValue_timeoutMS(value, budget) {
            let streak = self.timeouts.fetch_add(1, Ordering::AcqRel) + 1;
            let signaled = self.lifetime_event.signaledValue();
            let submitted = self.submitted.load(Ordering::Acquire);
            if streak >= CONSECUTIVE_EVENT_WAIT_TIMEOUTS {
                let error = format!(
                    "Metal 4 DeviceLost: {streak} consecutive SharedEvent wait budgets exceeded \
                     (last waited {budget} ms for lifetime event value {value}, reached {signaled}, \
                     submitted {submitted}); the runtime is poisoned and will not keep retrying"
                );
                self.poison.set(error.as_str());
                return Err(error);
            }
            return Err(event_wait_timeout_error(budget, value, signaled, submitted));
        }
        self.timeouts.store(0, Ordering::Release);
        self.checked_value().map(|_| ())
    }

    /// Wait until the GPU has completed commit `stamp`, for a host thread that
    /// wants to touch memory a command at `stamp` used.
    ///
    /// Checks poison first, treats `0` as "no GPU use", and returns at once
    /// when the (honesty-checked) lifetime event is already there. A `stamp`
    /// above [`Self::submitted`] is **not** waited on: the command is still in
    /// an open command buffer, no commit exists that would signal it, and the
    /// wait could only spend the whole budget. The caller flushes through the
    /// thread that owns the runtime (`commit(false)`) and asks again.
    pub(crate) fn wait_stamp(&self, stamp: u64) -> Result<(), HostAccessError> {
        self.ensure_not_poisoned().map_err(HostAccessError::Runtime)?;
        if stamp == 0 {
            return Ok(());
        }
        if self.checked_value().map_err(HostAccessError::Runtime)? >= stamp {
            return Ok(());
        }
        let submitted = self.submitted();
        if stamp > submitted {
            return Err(HostAccessError::NotSubmitted {
                needed: stamp,
                submitted,
            });
        }
        self.wait(stamp).map_err(HostAccessError::Runtime)
    }
}

impl ojas_core::Timeline for HostFence {
    fn submitted(&self) -> Result<u64, ojas_core::FenceError> {
        if self.ensure_not_poisoned().is_err() {
            return Err(ojas_core::FenceError::Poisoned);
        }
        Ok(self.submitted())
    }

    fn completed(&self) -> Result<u64, ojas_core::FenceError> {
        if self.ensure_not_poisoned().is_err() {
            return Err(ojas_core::FenceError::Poisoned);
        }
        self.checked_value().map_err(|_| ojas_core::FenceError::Poisoned)
    }

    fn current_value(&self) -> u64 {
        self.checked_value().unwrap_or(0)
    }

    fn wait(&self, value: u64) -> Result<(), ojas_core::FenceError> {
        match self.wait_stamp(value) {
            Ok(()) => Ok(()),
            Err(HostAccessError::NotSubmitted { needed, submitted }) => {
                Err(ojas_core::FenceError::NotSubmitted {
                    value: needed,
                    submitted,
                })
            }
            Err(HostAccessError::Busy(why)) => Err(ojas_core::FenceError::Busy { detail: why }),
            Err(HostAccessError::Runtime(why)) => {
                if why.contains("poisoned") || why.contains("DeviceLost") {
                    Err(ojas_core::FenceError::Poisoned)
                } else if why.contains("timeout") || why.contains("exceeded") {
                    Err(ojas_core::FenceError::TimedOut {
                        value,
                        completed: self.current_value(),
                    })
                } else {
                    Err(ojas_core::FenceError::Refused(ojas_core::OjasError::Backend {
                        id: ojas_core::BackendId::Metal,
                        detail: why,
                    }))
                }
            }
        }
    }
}

/// The runtime parts a [`HostFence`] is built from, borrowed so the runtime's
/// constructor needs no clones of its own.
pub(super) struct FenceParts<'a> {
    pub(super) lifetime_event: &'a Retained<ProtocolObject<dyn MTLSharedEvent>>,
    pub(super) submitted: &'a Arc<AtomicU64>,
    pub(super) wait_ms: &'a Arc<AtomicU64>,
    pub(super) timeouts: &'a Arc<AtomicU32>,
    pub(super) poison: &'a Arc<RuntimePoison>,
}

impl GpuRuntime {
    /// The thread-safe fence over this runtime's lifetime event.
    pub(crate) fn host_fence(&self) -> &HostFence {
        &self.fence
    }
}

/// A `HostFence` crosses threads with no `unsafe`: it holds a `Send + Sync`
/// event and `Arc`s of atomics. A `!Send` field stops this file compiling.
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    send_sync::<HostFence>();
};
