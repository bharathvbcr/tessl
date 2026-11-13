//! Which running executables [`super::others_running`] counts as another GPU user.
//!
//! Pure string logic, split out so it is testable without a process table.

/// Whether `path`, a running process's executable, is a test, bench or probe binary of a tessl
/// checkout.
///
/// Test binaries live in `target/<profile>/deps/<name>-<hash>` up to Cargo 1.99 and in
/// `target/<profile>/build/<pkg>/<hash>/out/<name>-<hash>` from 1.100 on. That `out/`
/// directory also holds the build script's own executable, which is not a GPU process.
pub fn is_tessl_gpu_binary(path: &str) -> bool {
    let in_target =
        path.contains("/research/tessl") && (path.contains("/target/release/") || path.contains("/target/debug/"));
    let file = path.rsplit('/').next().unwrap_or("");
    let in_build_out = path.contains("/build/")
        && path.contains("/out/")
        && !file.starts_with("build_script_")
        && !file.starts_with("build-script-");
    let gpu_bin = path.contains("/deps/")
        || in_build_out
        || file.starts_with("bench_")
        || file.starts_with("probe_")
        || file.starts_with("profile_");
    in_target && gpu_bin
}

#[cfg(test)]
mod tests {
    use super::is_tessl_gpu_binary;

    #[test]
    fn a_cargo_1_99_test_binary_in_deps_is_counted() {
        assert!(is_tessl_gpu_binary(
            "/Users/u/Code/research/tessl/target/release/deps/capi-0a1b2c3d4e5f6789"
        ));
    }

    #[test]
    fn a_cargo_1_100_test_binary_in_build_out_is_counted() {
        assert!(is_tessl_gpu_binary(
            "/Users/u/Code/research/tessl/target/release/build/tessl/0a1b2c3d4e5f6789/out/capi-0a1b2c3d4e5f6789"
        ));
        assert!(is_tessl_gpu_binary(
            "/Users/u/Code/research/tessl/target/debug/build/tessl/0a1b2c3d4e5f6789/out/gemm_ragged_shapes-9f8e7d6c5b4a3210"
        ));
    }

    #[test]
    fn a_build_script_executable_is_not_a_gpu_process() {
        assert!(!is_tessl_gpu_binary(
            "/Users/u/Code/research/tessl/target/release/build/tessl/0a1b2c3d4e5f6789/out/build_script_build-0a1b2c3d4e5f6789"
        ));
        assert!(!is_tessl_gpu_binary(
            "/Users/u/Code/research/tessl/target/release/build/tessl/0a1b2c3d4e5f6789/out/build-script-build"
        ));
    }

    #[test]
    fn named_bench_probe_and_profile_bins_are_counted_and_unrelated_paths_are_not() {
        assert!(is_tessl_gpu_binary(
            "/Users/u/Code/research/tessl/target/release/bench_gemm"
        ));
        assert!(is_tessl_gpu_binary(
            "/Users/u/Code/research/tessl-w8/target/release/probe_gemm_parity"
        ));
        assert!(is_tessl_gpu_binary(
            "/Users/u/Code/research/tessl/target/release/profile_embedgemma2"
        ));
        assert!(!is_tessl_gpu_binary("/usr/bin/python3"));
        assert!(!is_tessl_gpu_binary(
            "/Users/u/Code/research/ojas/target/release/deps/gemm-0a1b2c3d4e5f6789"
        ));
        assert!(!is_tessl_gpu_binary(
            "/Users/u/Code/research/tessl/target/release/tessl_unrelated_tool"
        ));
    }
}
