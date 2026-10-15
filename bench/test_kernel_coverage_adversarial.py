#!/usr/bin/env python3
"""Adversarial regressions for `kernel_coverage.py` (wave-7 lane gp/w7-kcov).

Each test pins one way the source scan or the CLI misreports. Tests whose name
ends in `` FAIL against the implementation as committed: they are
the findings, not a green gate. Run with:

  python3 -I bench/test_kernel_coverage_adversarial.py
"""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel_coverage  # noqa: E402

STAMP = "#define STAMP(NAME, T) \\\nkernel void NAME(device T* x) {}\n"


def scan(text):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "a.metal"), "w") as f:
        f.write(text)
    names, _ = kernel_coverage.inventory(d, metallib="/nonexistent")
    return names


class PhantomKernels(unittest.TestCase):
    """Names the scan reports that no metallib can export."""

    def test_invocation_in_block_comment_is_not_a_kernel(self):
        self.assertEqual(scan(STAMP + "STAMP(live, float)\n/*\nSTAMP(dead, float)\n*/\n"), {"live"})

    def test_invocation_under_if_0_is_not_a_kernel(self):
        self.assertEqual(scan(STAMP + "STAMP(live, float)\n#if 0\nSTAMP(dead, float)\n#endif\n"), {"live"})

    def test_invocation_under_unset_ifdef_is_not_a_kernel(self):
        self.assertEqual(scan(STAMP + "STAMP(live, float)\n#ifdef NOPE\nSTAMP(dead, float)\n#endif\n"), {"live"})

    def test_kernel_word_inside_a_macro_body_comment_is_not_a_kernel(self):
        body = ("#define S2(NAME) \\\n  /* kernel void ghost(x) */ \\\n"
                " kernel void NAME(device float* x) {}\nS2(real)\n")
        self.assertEqual(scan(body), {"real"})

    def test_plain_kernel_in_block_comment_is_not_a_kernel(self):
        self.assertEqual(scan("kernel void live(device float* x) {}\n"
                              "/*\nkernel void dead(device float* x) {}\n*/\n"), {"live"})


class MissedKernels(unittest.TestCase):
    """Real kernels the scan cannot see (only the metallib cross-check saves it)."""

    def test_two_invocations_on_one_line(self):
        self.assertEqual(scan(STAMP + "STAMP(a1, float) STAMP(a2, float)\n"), {"a1", "a2"})

    def test_indented_invocation(self):
        self.assertEqual(scan(STAMP + "STAMP(top, float)\n  STAMP(indented, float)\n"), {"top", "indented"})

    def test_indented_plain_kernel(self):
        self.assertEqual(scan("kernel void top(device float* x) {}\n"
                              "  kernel void indented(device float* x) {}\n"), {"top", "indented"})

    def test_host_name_alias_is_the_exported_name(self):
        # metal-nm exports the host_name, not the C++ symbol.
        self.assertEqual(scan('[[host_name("alias")]] kernel void inner(device float* x) {}\n'),
                         {"alias"})


class CliExitStatus(unittest.TestCase):
    """--check: a check that could not run must differ from one that found gaps."""

    def run_cli(self, *argv, env=None):
        return subprocess.run([sys.executable, "-I", os.path.join(HERE, "kernel_coverage.py"), *argv],
                              capture_output=True, text=True, timeout=120, env=env)

    def test_no_binary_in_cargo_target_dir_cannot_run(self):
        # CARGO_TARGET_DIR is honoured: an empty one has no bench binary.
        empty = tempfile.mkdtemp()
        r = self.run_cli("--check", env=dict(os.environ, CARGO_TARGET_DIR=empty))
        self.assertEqual(r.returncode, kernel_coverage.EXIT_CANNOT_RUN, r.stderr)
        self.assertIn(os.path.join(empty, "release"), r.stderr)

    def test_missing_toolchain_cannot_run_and_never_falls_back_to_a_scan(self):
        r = self.run_cli("--check", env=dict(os.environ, PATH="/nonexistent"))
        self.assertEqual(r.returncode, kernel_coverage.EXIT_CANNOT_RUN, r.stderr)
        self.assertIn("xcrun metal", r.stderr)

    def test_unrunnable_check_has_its_own_exit_status(self):
        # A missing explicit metallib: the check cannot run at all.
        cannot_run = self.run_cli("--check", "--metallib", "/nonexistent/x.metallib")
        self.assertEqual(cannot_run.returncode, kernel_coverage.EXIT_CANNOT_RUN, cannot_run.stderr)
        self.assertNotEqual(cannot_run.returncode, 0, "must never report success")
        # Gaps found is SystemExit(<message>) == 1; the unrunnable case must not
        # share that status, or a CI wrapper cannot tell "uncovered" from "broken".
        self.assertNotEqual(cannot_run.returncode, 1,
                            "could-not-run shares exit 1 with found-gaps; use a distinct status (the repo uses 77 for SKIPPED, 3 for unstable)")


class Freshness(unittest.TestCase):
    """A bench binary older than what it is built from is refused."""

    def make_crate(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "kernels"))
        for rel in ("build.rs", "kernels/a.metal", "kernels/a.h"):
            open(os.path.join(d, rel), "w").write("x")
        binary = os.path.join(d, "bin")
        open(binary, "w").write("x")
        return d, binary

    def test_binary_older_than_a_kernel_source_is_refused(self):
        d, binary = self.make_crate()
        os.utime(binary, (1000, 1000))
        for rel in ("build.rs", "kernels/a.metal", "kernels/a.h"):
            for other in ("build.rs", "kernels/a.metal", "kernels/a.h"):
                os.utime(os.path.join(d, other), (500, 500))
            os.utime(os.path.join(d, rel), (2000, 2000))
            with self.assertRaises(kernel_coverage.CannotRun) as cm:
                kernel_coverage.require_fresh(binary, d)
            self.assertIn("cargo build --release --bins", cm.exception.message, rel)

    def test_binary_newer_than_every_input_is_accepted(self):
        d, binary = self.make_crate()
        for rel in ("build.rs", "kernels/a.metal", "kernels/a.h"):
            os.utime(os.path.join(d, rel), (500, 500))
        os.utime(binary, (1000, 1000))
        kernel_coverage.require_fresh(binary, d)


class Census(unittest.TestCase):
    def test_template_definition_is_not_an_entry_point_but_its_instantiation_is(self):
        src = ("template <typename T> kernel void gen(device T* x) {}\n"
               'template [[host_name("gen_f32")]] kernel void gen<float>(device float*);\n')
        self.assertEqual(scan(src), {"gen_f32"})

    def test_instantiation_without_host_name_is_refused(self):
        with self.assertRaises(SystemExit) as cm:
            scan("template <typename T> kernel void gen(device T* x) {}\n"
                 "template kernel void gen<float>(device float*);\n")
        self.assertIn("host_name", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
