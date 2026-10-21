//! This process's peak physical footprint. The one owner: the benchmark bins
//! reach it through `common`, and the integration tests that print their own
//! peak include this file with `#[path]`. Its only dependency is `libc`.

#![allow(dead_code)] // Each includer uses a subset.

/// This process's lifetime peak physical footprint in bytes: the figure
/// `/usr/bin/time -l` prints as `peak memory footprint`. Not `ru_maxrss`,
/// which under-reads when macOS compresses or reclaims pages
/// (bench/results/qwen35_load_rss_m5pro.txt saw 8.07 GB against 10.79 GB).
pub fn peak_footprint() -> Result<u64, String> {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: `proc_pid_rusage` fills a `rusage_info_v4` for this flavor, and
    // `info` is one, zero-initialised.
    let rc = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V4,
            info.as_mut_ptr().cast::<libc::rusage_info_t>(),
        )
    };
    if rc != 0 {
        return Err(format!("proc_pid_rusage: {}", std::io::Error::last_os_error()));
    }
    // SAFETY: zero-initialised and then filled by a successful call.
    Ok(unsafe { info.assume_init() }.ri_lifetime_max_phys_footprint)
}
