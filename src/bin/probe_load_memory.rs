//! Peak host memory of one Qwen3.5-2B load, nothing else.
//!
//! ```text
//! QWEN35_2B_SAFETENSORS=... cargo run --release --bin probe_load_memory -- --bf16
//! QWEN35_2B_SAFETENSORS=... cargo run --release --bin probe_load_memory -- --f32
//! QWEN35_2B_SAFETENSORS=... cargo run --release --bin probe_load_memory -- --bf16 --tower
//! ```
//!
//! Opens the checkpoint, runs [`Qwen35Model::load`] (or
//! [`Qwen35Model::load_tower`] with `--tower`) at one precision, and prints
//! the process's peak resident set (`getrusage` `ru_maxrss`) before and after
//! the load, beside `MTLDevice::currentAllocatedSize`. Shared-storage weights
//! the loader writes are resident too, so the device bytes are part of the
//! peak; what the peak holds beyond them is the loader's host temporaries.
//! One load per process, because `ru_maxrss` never comes back down.

use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

type Res<T> = Result<T, String>;

const GB: f64 = 1e9;

fn gb(b: u64) -> String {
    format!("{:.2} GB", b as f64 / GB)
}

/// This process's peak resident set so far, in bytes (`ru_maxrss` is bytes
/// on Darwin, the only target tessl builds for).
fn peak_rss() -> Res<u64> {
    let mut u = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `getrusage` fills the struct it is handed; RUSAGE_SELF is valid.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, u.as_mut_ptr()) } != 0 {
        return Err(format!("getrusage: {}", std::io::Error::last_os_error()));
    }
    // SAFETY: zero-initialised and then filled by a successful `getrusage`.
    let maxrss = unsafe { u.assume_init() }.ru_maxrss;
    u64::try_from(maxrss).map_err(|_| format!("getrusage: negative ru_maxrss {maxrss}"))
}

fn main() -> Res<()> {
    let path = std::env::var("QWEN35_2B_SAFETENSORS")
        .map_err(|_| "set QWEN35_2B_SAFETENSORS to the Qwen3.5-2B .safetensors".to_string())?;
    let (mut precision, mut with_head) = (None, true);
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--bf16" => precision = Some(Precision::Bf16),
            "--f32" => precision = Some(Precision::F32),
            "--tower" => with_head = false,
            _ => return Err(format!("expected --bf16 or --f32, and optionally --tower; got {arg:?}")),
        }
    }
    let precision = precision.ok_or("pass --bf16 or --f32")?;

    let rt = GpuRuntime::new()?;
    let st = SafeTensors::open(std::path::Path::new(&path))?;
    let before = peak_rss()?;
    let dev0 = rt.current_allocated_bytes();
    let cfg = Qwen35Config::qwen35_2b()?;
    let prefix = "model.language_model.";
    let model = if with_head {
        Qwen35Model::load(&rt, &st, prefix, cfg, precision)?
    } else {
        Qwen35Model::load_tower(&rt, &st, prefix, cfg, precision)?
    };
    rt.synchronize()?;
    let after = peak_rss()?;
    let device = rt.current_allocated_bytes() - dev0;
    println!(
        "{precision:?}{}: peak rss {} before load, {} after (+{}); device +{}; \
         peak beyond the runtime and the device weights {}",
        if with_head { "" } else { " tower" },
        gb(before),
        gb(after),
        gb(after - before),
        gb(device),
        gb((after - before).saturating_sub(device)),
    );
    drop(model);
    Ok(())
}
