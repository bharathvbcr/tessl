//! Partition of the built metallib: every text symbol is checked or excluded.

use std::collections::{BTreeMap, BTreeSet};

use tessl::GpuRuntime;

pub struct Ledger {
    names: BTreeSet<String>,
    checked: BTreeSet<String>,
    excluded: BTreeMap<String, String>,
}

impl Ledger {
    pub fn new() -> Self {
        let nm = std::process::Command::new("xcrun")
            .args(["metal-nm", tessl::metallib_path()])
            .output()
            .expect("xcrun metal-nm");
        assert!(
            nm.status.success(),
            "metal-nm failed: {}",
            String::from_utf8_lossy(&nm.stderr)
        );
        let names = String::from_utf8_lossy(&nm.stdout)
            .lines()
            .filter(|line| line.split_whitespace().nth(1) == Some("T"))
            .map(|line| {
                line.split_whitespace()
                    .nth(2)
                    .unwrap_or_else(|| panic!("metal-nm T line has no name: {line}"))
                    .to_string()
            })
            .collect();
        Self {
            names,
            checked: BTreeSet::new(),
            excluded: BTreeMap::new(),
        }
    }

    pub fn exclude(&mut self, name: &str, reason: &str) {
        assert!(
            self.names.contains(name),
            "excluding {name}, which metal-nm did not list"
        );
        assert!(
            self.excluded.insert(name.to_string(), reason.to_string()).is_none(),
            "excluded {name} twice"
        );
    }

    /// Run `body`, which must dispatch and compare. Credit `expected` only when
    /// the per-dispatch stamps show a dispatch of each name inside `body`.
    ///
    /// A pipeline request alone is not evidence: an op may request a kernel and
    /// never dispatch it, and the global dispatch count cannot say which kernel
    /// ran. The runtime stamps every dispatch with the kernel it bound
    /// ([`tessl::dispatch_stamps`]), so the credit needs a per-name dispatch
    /// count as well as the request the trace saw. The stamp heap exists only on a runtime built with
    /// `tessl::ab_flags::set_dispatch_stamps(true)`; without it nothing can be
    /// credited and this panics rather than crediting on weaker evidence. A
    /// stamp batch that overflowed, dropped a command buffer or failed to
    /// resolve is a panic too, never a partial credit.
    pub fn credit<S: AsRef<str>>(&mut self, rt: &GpuRuntime, label: &str, expected: &[S], body: impl FnOnce()) {
        assert!(!expected.is_empty(), "{label}: no kernels to credit");
        for name in expected {
            let name = name.as_ref();
            assert!(
                self.names.contains(name),
                "{label}: {name} is not a metallib text symbol"
            );
            assert!(
                !self.excluded.contains_key(name),
                "{label}: {name} is excluded and must not be dispatched"
            );
        }
        assert!(
            tessl::ab_flags::dispatch_stamps() && rt.dispatch_stamp_heap_available(),
            "{label}: per-dispatch stamps are off, so no dispatch can be attributed to a kernel; \
             the runtime must be built after set_dispatch_stamps(true)"
        );
        // Drain whatever an earlier body left queued, so the window below holds
        // this body's dispatches only.
        rt.synchronize()
            .unwrap_or_else(|e| panic!("{label}: pre-body sync: {e}"));
        let before: Vec<u64> = expected
            .iter()
            .map(|n| tessl::runtime::traced_kernel_count(n.as_ref()))
            .collect();
        let _stale = rt.take_dispatch_stamps();
        let _reset = rt.take_dispatch_count();
        body();
        rt.synchronize()
            .unwrap_or_else(|e| panic!("{label}: post-body sync: {e}"));
        let dispatches = rt.take_dispatch_count();
        assert!(dispatches > 0, "{label}: dispatched nothing");
        let batch = rt
            .take_dispatch_stamps()
            .unwrap_or_else(|| panic!("{label}: {dispatches} dispatches but no stamp batch resolved"))
            .unwrap_or_else(|e| panic!("{label}: stamps could not be read back: {e}"));
        assert!(
            !batch.overflowed && batch.dropped_batches == 0,
            "{label}: stamp batch is incomplete (overflowed {}, dropped command buffers {}); \
             a prefix cannot credit a kernel",
            batch.overflowed,
            batch.dropped_batches
        );
        let mut ran: BTreeMap<&str, usize> = BTreeMap::new();
        for stamp in &batch.stamps {
            *ran.entry(stamp.name.as_str()).or_default() += 1;
        }
        for (name, &prior) in expected.iter().zip(before.iter()) {
            let name = name.as_ref();
            let after = tessl::runtime::traced_kernel_count(name);
            assert!(
                after > prior,
                "{label}: {name} was not requested (trace {prior} -> {after}, dispatches {dispatches})"
            );
            assert!(
                ran.get(name).copied().unwrap_or(0) > 0,
                "{label}: {name} never dispatched (stamped dispatches by kernel: {ran:?}, global count {dispatches})"
            );
            self.checked.insert(name.to_string());
        }
    }

    pub fn report_and_assert(&self) {
        println!("metal-nm {}", self.names.len());
        println!("checked {}", self.checked.len());
        println!("excluded {}", self.excluded.len());
        for name in &self.checked {
            println!("PASS {name}");
        }
        for (name, reason) in &self.excluded {
            println!("EXCLUDE {name} | {reason}");
        }
        let mut broken = Vec::new();
        for name in &self.names {
            let in_checked = self.checked.contains(name);
            let in_excluded = self.excluded.contains_key(name);
            if in_checked == in_excluded {
                broken.push(name.clone());
            }
        }
        assert!(
            broken.is_empty(),
            "checked {} + excluded {} != metal-nm {}: {broken:?}",
            self.checked.len(),
            self.excluded.len(),
            self.names.len()
        );
        assert_eq!(self.checked.len() + self.excluded.len(), self.names.len());
    }
}
