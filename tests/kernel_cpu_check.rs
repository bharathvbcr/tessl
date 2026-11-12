//! Every numeric kernel in the built metallib, against an independent CPU reference.
//!
//! Non-numeric binding probes are excluded by name. Checked plus excluded equals
//! the `metal-nm` text-symbol count. The sweep needs the kernel trace to tell
//! which kernel ran, so the test re-runs itself in a child process with
//! `TESSL_KERNEL_TRACE=1` (`common::in_traced_child`) and prints the child's
//! report, which a plain `cargo test` run therefore exercises too.

mod common;
mod kcheck;

use kcheck::Ledger;

#[test]
fn every_numeric_kernel_matches_a_cpu_reference() {
    let report = common::in_traced_child("every_numeric_kernel_matches_a_cpu_reference", || {
        assert!(
            tessl::runtime::kernel_trace_enabled(),
            "TESSL_KERNEL_TRACE is off; the sweep cannot tell which kernel ran"
        );
        common::with_gpu(|rt| {
            let mut ledger = Ledger::new();
            // Binding probes. The only caller is a quant-prep test; the default
            // binary has no numeric op that launches them. kernels/mtl_tensor.metal
            // says they exist so the MTLTensor binding path is exercised by a write
            // then a read.
            ledger.exclude(
                "mtl_tensor_fill_i8",
                "MTLTensor binding probe, not a numeric op; host launch is quant-prep only",
            );
            ledger.exclude(
                "mtl_tensor_read_i8",
                "MTLTensor binding probe, not a numeric op; host launch is quant-prep only",
            );
            kcheck::flash::run(rt, &mut ledger);
            kcheck::gemm::run(rt, &mut ledger);
            kcheck::quant::run(rt, &mut ledger);
            kcheck::rest::run(rt, &mut ledger);
            kcheck::adamw::run(rt, &mut ledger);
            kcheck::qwen::run(rt, &mut ledger);
            kcheck::emeans::run(rt, &mut ledger);
            ledger.report_and_assert();
        });
    });
    if let Some(report) = report {
        println!("{report}");
    }
}
