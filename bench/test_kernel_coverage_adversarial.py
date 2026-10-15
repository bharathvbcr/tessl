#!/usr/bin/env python3
"""Adversarial regressions for `kernel_coverage.py` (wave-7 lane gp/w7-kcov).

Each test pins one way the source scan or the CLI misreports. Tests whose name
ends in `_KNOWN_BREAK` FAIL against the implementation as committed: they are
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

    def test_invocation_in_block_comment_is_not_a_kernel_KNOWN_BREAK(self):
        self.assertEqual(scan(STAMP + "STAMP(live, float)\n/*\nSTAMP(dead, float)\n*/\n"), {"live"})

    def test_invocation_under_if_0_is_not_a_kernel_KNOWN_BREAK(self):
        self.assertEqual(scan(STAMP + "STAMP(live, float)\n#if 0\nSTAMP(dead, float)\n#endif\n"), {"live"})

    def test_invocation_under_unset_ifdef_is_not_a_kernel_KNOWN_BREAK(self):
        self.assertEqual(scan(STAMP + "STAMP(live, float)\n#ifdef NOPE\nSTAMP(dead, float)\n#endif\n"), {"live"})

    def test_kernel_word_inside_a_macro_body_comment_is_not_a_kernel_KNOWN_BREAK(self):
        body = ("#define S2(NAME) \\\n  /* kernel void ghost(x) */ \\\n"
                " kernel void NAME(device float* x) {}\nS2(real)\n")
        self.assertEqual(scan(body), {"real"})

    def test_plain_kernel_in_block_comment_is_not_a_kernel_KNOWN_BREAK(self):
        self.assertEqual(scan("kernel void live(device float* x) {}\n"
                              "/*\nkernel void dead(device float* x) {}\n*/\n"), {"live"})


class MissedKernels(unittest.TestCase):
    """Real kernels the scan cannot see (only the metallib cross-check saves it)."""

    def test_two_invocations_on_one_line_KNOWN_BREAK(self):
        self.assertEqual(scan(STAMP + "STAMP(a1, float) STAMP(a2, float)\n"), {"a1", "a2"})

    def test_indented_invocation_KNOWN_BREAK(self):
        self.assertEqual(scan(STAMP + "STAMP(top, float)\n  STAMP(indented, float)\n"), {"top", "indented"})

    def test_indented_plain_kernel_KNOWN_BREAK(self):
        self.assertEqual(scan("kernel void top(device float* x) {}\n"
                              "  kernel void indented(device float* x) {}\n"), {"top", "indented"})

    def test_host_name_alias_is_the_exported_name_KNOWN_BREAK(self):
        # metal-nm exports the host_name, not the C++ symbol.
        self.assertEqual(scan('[[host_name("alias")]] kernel void inner(device float* x) {}\n'),
                         {"alias"})


class CliExitStatus(unittest.TestCase):
    """--check: a check that could not run must differ from one that found gaps."""

    def run_cli(self, *argv):
        return subprocess.run([sys.executable, "-I", os.path.join(HERE, "kernel_coverage.py"), *argv],
                              capture_output=True, text=True, timeout=120)

    def test_unrunnable_check_has_its_own_exit_status_KNOWN_BREAK(self):
        # A missing explicit metallib: the check cannot run at all.
        cannot_run = self.run_cli("--check", "--metallib", "/nonexistent/x.metallib")
        self.assertNotEqual(cannot_run.returncode, 0, "must never report success")
        # Gaps found is SystemExit(<message>) == 1; the unrunnable case must not
        # share that status, or a CI wrapper cannot tell "uncovered" from "broken".
        self.assertNotEqual(cannot_run.returncode, 1,
                            "could-not-run shares exit 1 with found-gaps; use a distinct status (the repo uses 77 for SKIPPED, 3 for unstable)")


if __name__ == "__main__":
    unittest.main()
