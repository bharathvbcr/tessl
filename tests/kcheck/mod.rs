//! Device sweep: every numeric metallib entry compared to a CPU reference.

mod ledger;

pub mod adamw;
pub mod emeans;
pub mod flash;
pub mod gemm;
pub mod quant;
pub mod qwen;
pub mod rest;

pub use ledger::Ledger;
