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
    /// each name's pipeline request count grew and a dispatch happened.
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
        let before: Vec<u64> = expected
            .iter()
            .map(|n| tessl::runtime::traced_kernel_count(n.as_ref()))
            .collect();
        let _reset = rt.take_dispatch_count();
        body();
        let dispatches = rt.take_dispatch_count();
        assert!(dispatches > 0, "{label}: dispatched nothing");
        for (name, &prior) in expected.iter().zip(before.iter()) {
            let name = name.as_ref();
            let after = tessl::runtime::traced_kernel_count(name);
            assert!(
                after > prior,
                "{label}: {name} was not requested (trace {prior} -> {after}, dispatches {dispatches})"
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
