#!/usr/bin/env python3
"""Shared, dependency-free evidence helpers for cross-runtime benchmarks."""

from __future__ import annotations

import datetime as _datetime
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import uuid


DEFAULT_MAX_RATIO_SPREAD = 1.10
MAX_RATIO_SPREAD = 1.25
DEFAULT_OUTER_ROUNDS = 6
MIN_OUTER_ROUNDS = 4
MIN_INNER_ITERS = 3
ENVIRONMENT_PROBE_POLICY = (
    "thermal/power/load are provenance-only: pmset is macOS-specific and may emit "
    "non-numeric state, while normalized host load is workload- and CPU-count-dependent; "
    "paired ratio spread is the portable enforced stability gate"
)
BENCHMARK_ENV_PREFIXES = (
    "BENCH_",
    "TESSL_",
    "METAL_",
    "MTL_",
    "MLX_",
    "PYTORCH_",
    "OMP_",
    "MKL_",
    "VECLIB_",
    "ACCELERATE_",
)

# macOS PATH_MAX is below this, while the explicit bound prevents a malformed
# or unrelated binary from turning suffix discovery into an unbounded reverse
# scan. Cargo/Rust environment string literals are UTF-8, so non-UTF-8 paths
# cannot be embedded by `env!` and are deliberately outside this contract.
MAX_EMBEDDED_METALLIB_PATH_BYTES = 4096
MAX_EMBEDDED_METALLIB_MARKERS = 1024
MAX_EMBEDDED_METALLIB_PATH_PROBES = 16384


def parse_ratio_spread_limit(text: str) -> float:
    """Parse a finite, explicitly bounded max/min ratio-spread limit."""
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"expected a number, got {text!r}") from exc
    if not math.isfinite(value) or not 1.0 <= value <= MAX_RATIO_SPREAD:
        raise ValueError(
            f"ratio-spread limit must be finite and in [1, {MAX_RATIO_SPREAD:g}], got {text!r}"
        )
    return value


def validate_evidence_sample_counts(rounds: int, iters: int) -> None:
    if rounds < MIN_OUTER_ROUNDS or rounds % 2 != 0:
        raise ValueError(
            f"rounds must be an even integer >= {MIN_OUTER_ROUNDS} for exact AB/BA "
            f"balance, a paired median, and drift estimation; got {rounds}"
        )
    if iters < MIN_INNER_ITERS:
        raise ValueError(
            f"iters must be >= {MIN_INNER_ITERS} for each child-process median"
        )


def requested_output_path(argv: list[str]) -> str | None:
    """Find argparse's last usable ``--out`` value before validation runs."""
    selected = None
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--out":
            if index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                selected = argv[index + 1]
                index += 2
                continue
        elif token.startswith("--out=") and token != "--out=":
            selected = token.split("=", 1)[1]
        index += 1
    return selected


def clean_benchmark_env(overrides: dict[str, str]) -> dict[str, str]:
    """Drop inherited benchmark/tuning knobs, then install explicit overrides."""
    clean = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(BENCHMARK_ENV_PREFIXES)
    }
    clean.update(overrides)
    return clean


def geometric_mean(values: list[float], *, label: str) -> float:
    checked = _positive_finite(values, label=label)
    return math.exp(sum(math.log(value) for value in checked) / len(checked))


def series_summary(values: list[float], *, label: str) -> dict:
    """Return robust location plus every outer-round aggregate value."""
    import statistics

    checked = _positive_finite(values, label=label)
    minimum = min(checked)
    maximum = max(checked)
    return {
        "median": statistics.median(checked),
        "min": minimum,
        "max": maximum,
        "spread": maximum / minimum,
        "outer_round_values": checked,
    }


def enforce_ratio_spread(summary: dict, *, limit: float, label: str) -> None:
    if not math.isfinite(limit) or not 1.0 <= limit <= MAX_RATIO_SPREAD:
        raise ValueError(
            f"{label}: ratio-spread limit must be in [1, {MAX_RATIO_SPREAD:g}]"
        )
    spread = summary.get("spread")
    if not isinstance(spread, (int, float)) or not math.isfinite(spread):
        raise ValueError(f"{label}: missing/non-finite ratio spread")
    if spread > limit:
        raise ValueError(
            f"{label}: paired ratio spread {spread:.3f}x exceeds bounded limit {limit:.3f}x"
        )


def start_provenance(
    *,
    driver_path: str,
    argv: list[str],
    repo_scope: str,
    executable_inputs: dict[str, str],
    benchmark_config: dict,
    probe_device: bool = True,
) -> dict:
    """Capture the run inputs and environment without relying on optional packages."""
    driver_path = os.path.abspath(driver_path)
    invocation = _redact_argv([sys.executable, driver_path, *argv[1:]])
    git = _git_snapshot(repo_scope)
    inputs = {
        "driver": _file_record(driver_path),
        **{name: _file_record(path) for name, path in executable_inputs.items()},
    }
    provenance = {
        "schema_version": 1,
        "started_at_utc": _utc_now(),
        "invocation_argv": invocation,
        "invocation_shell": _shell_join(invocation),
        "cwd": _display_path(os.getcwd()),
        "path_policy": (
            "paths beneath the invoking user's home are recorded as $HOME-relative; "
            "hashes are computed from the real files"
        ),
        "environment_policy": (
            "full process environment not captured; inherited benchmark/tuning prefixes "
            f"{BENCHMARK_ENV_PREFIXES} are cleared and benchmark_config records only "
            "the explicit child overrides"
        ),
        "benchmark_config": _redact_public(benchmark_config),
        "git": git,
        "host": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "hardware_model": _sysctl_text("hw.model"),
            "memory_bytes": _sysctl_integer("hw.memsize"),
        },
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "mac_ver": platform.mac_ver()[0],
            "sw_vers": _command_record(["sw_vers"]),
        },
        "runtime": {
            "python": sys.version,
            "python_executable": _display_path(sys.executable),
            "packages": {
                name: _package_version(name) for name in ("numpy", "torch", "mlx")
            },
            "rustc": _command_record(["rustc", "--version"]),
        },
        "power": {
            "thermal_pressure": _command_record(["pmset", "-g", "therm"]),
            "power_source": _power_source_record(),
        },
        "load": {"start": _load_snapshot()},
        "inputs": inputs,
    }
    provenance["device"] = (
        _device_snapshot() if probe_device else {"probe": "disabled_for_test"}
    )
    if not (
        git.get("available")
        and git.get("revision")
        and isinstance(git.get("dirty"), bool)
        and git.get("tracked_diff_sha256")
    ):
        raise ValueError(f"Git revision/dirty/diff provenance unavailable: {git}")
    missing_hashes = [name for name, record in inputs.items() if not record.get("sha256")]
    if missing_hashes:
        raise ValueError(f"executable input hashes unavailable: {missing_hashes}")
    return provenance


def start_required_provenance(**kwargs) -> dict:
    try:
        return start_provenance(**kwargs)
    except ValueError as exc:
        raise SystemExit(f"cannot establish reproducible benchmark provenance: {exc}")


def finish_provenance(provenance: dict) -> None:
    provenance["finished_at_utc"] = _utc_now()
    provenance.setdefault("load", {})["finish"] = _load_snapshot()
    provenance.setdefault("power", {})["thermal_pressure_finish"] = _command_record(
        ["pmset", "-g", "therm"]
    )
    provenance.setdefault("power", {})["power_source_finish"] = _power_source_record()


def embedded_metallib_path(binary_path: str) -> str:
    """Resolve the bounded absolute UTF-8 metallib path compiled into a binary.

    A POSIX path component may contain every byte except NUL and ``/``. Do not
    guess a filename alphabet: locate each ``.metallib`` suffix, consider the
    bounded slash-started substrings ending there, and accept only an existing
    file. Multiple distinct live candidates fail closed rather than selecting
    whichever byte string happened to appear first in the executable.
    """
    try:
        with open(binary_path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise ValueError(f"cannot inspect benchmark binary {binary_path}: {exc}") from exc

    suffix = b".metallib"
    marker_count = data.count(suffix)
    if marker_count > MAX_EMBEDDED_METALLIB_MARKERS:
        raise ValueError(
            f"embedded metallib scan found {marker_count} suffix markers, above the "
            f"bounded limit {MAX_EMBEDDED_METALLIB_MARKERS}"
        )

    candidates = set()
    probes = 0
    cursor = 0
    while True:
        suffix_start = data.find(suffix, cursor)
        if suffix_start < 0:
            break
        suffix_end = suffix_start + len(suffix)
        window_start = max(0, suffix_end - MAX_EMBEDDED_METALLIB_PATH_BYTES)
        slash = data.find(b"/", window_start, suffix_start + 1)
        while slash >= 0:
            probes += 1
            if probes > MAX_EMBEDDED_METALLIB_PATH_PROBES:
                raise ValueError(
                    "embedded metallib path scan exceeded the bounded candidate-probe "
                    f"limit {MAX_EMBEDDED_METALLIB_PATH_PROBES}"
                )
            raw = data[slash:suffix_end]
            try:
                path = raw.decode("utf-8", errors="strict")
                if os.path.isfile(path):
                    candidates.add(os.path.realpath(path))
            except (OSError, ValueError, UnicodeError):
                # Most slash-started substrings are not paths; malformed UTF-8,
                # embedded NULs, overlong components, and inaccessible names are
                # ordinary rejected candidates, not alternate interpretations.
                pass
            slash = data.find(b"/", slash + 1, suffix_start + 1)
        cursor = suffix_end

    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one existing metallib path embedded in {binary_path}, "
            f"found {sorted(candidates)}"
        )
    return candidates.pop()


def atomic_write_json(path: str, payload: dict) -> None:
    """Durably replace a JSON artifact without exposing partial contents."""
    absolute = os.path.abspath(path)
    directory = os.path.dirname(absolute)
    fd, staged = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, absolute)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(staged)
        except FileNotFoundError:
            pass
        raise


class EvidenceOutput:
    """Atomic publication plus a fail-closed marker for the latest attempt.

    An older valid output is deliberately preserved if a rerun fails. The
    sidecar starts in ``not_published`` state, so that preserved file cannot be
    mistaken for evidence from the newer attempt (including after a crash).
    """

    def __init__(self, output_path: str | None, *, driver_path: str, argv: list[str]):
        self.output_path = os.path.abspath(output_path) if output_path else None
        self.marker_path = f"{self.output_path}.attempt.json" if self.output_path else None
        invocation = _redact_argv(
            [sys.executable, os.path.abspath(driver_path), *argv[1:]]
        )
        self.record = {
            "schema_version": 1,
            "attempt_id": str(uuid.uuid4()),
            "status": "not_published",
            "started_at_utc": _utc_now(),
            "invocation_argv": invocation,
            "invocation_shell": _shell_join(invocation),
            "output_path": _display_path(self.output_path) if self.output_path else None,
            "prior_output": (
                _file_record(self.output_path)
                if self.output_path and os.path.exists(self.output_path)
                else None
            ),
            "interpretation": (
                "Unless status is published, output_path does not represent this attempt."
            ),
        }

    def begin(self) -> None:
        if self.marker_path:
            atomic_write_json(self.marker_path, self.record)

    def publish(self, payload: dict) -> None:
        if not self.output_path or not self.marker_path:
            raise ValueError("cannot publish without an output path")
        atomic_write_json(self.output_path, payload)
        self.record.update({
            "status": "published",
            "finished_at_utc": _utc_now(),
            "published_output": _file_record(self.output_path),
        })
        atomic_write_json(self.marker_path, self.record)


def _positive_finite(values: list[float], *, label: str) -> list[float]:
    if not values:
        raise ValueError(f"{label}: no samples")
    checked = []
    for value in values:
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{label}: expected positive finite sample, got {value!r}")
        checked.append(number)
    return checked


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _redact_text(value: str) -> str:
    home = os.path.abspath(os.path.expanduser("~"))
    if home != os.path.sep:
        return value.replace(home, "$HOME")
    return value


def _redact_argv(argv: list[str]) -> list[str]:
    return [_redact_text(value) for value in argv]


def _redact_public(value: object) -> object:
    """Recursively redact the local home prefix from publishable provenance."""
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [_redact_public(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_public(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_public(item) for key, item in value.items()}
    return value


def _display_path(path: str) -> str:
    return _redact_text(os.path.abspath(path))


def _shell_join(argv: list[str]) -> str:
    import shlex

    return shlex.join(argv)


def _load_snapshot() -> dict:
    try:
        one, five, fifteen = os.getloadavg()
        cpu_count = os.cpu_count()
        return {
            "loadavg_1m": one,
            "loadavg_5m": five,
            "loadavg_15m": fifteen,
            "cpu_count_normalizer": cpu_count,
            "normalized_loadavg_1m": one / cpu_count if cpu_count else None,
            "normalized_loadavg_5m": five / cpu_count if cpu_count else None,
            "normalized_loadavg_15m": fifteen / cpu_count if cpu_count else None,
        }
    except OSError as exc:
        return {"unavailable": str(exc)}


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _file_record(path: str) -> dict:
    absolute = os.path.abspath(path)
    try:
        stat = os.stat(absolute)
        digest = hashlib.sha256()
        with open(absolute, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path": _display_path(absolute),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest(),
        }
    except OSError as exc:
        return {"path": _display_path(absolute), "unavailable": _redact_public(str(exc))}


def _run_raw(argv: list[str], *, cwd: str | None = None, timeout: float = 10.0) -> dict:
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "argv": argv,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": str(exc), "stdout": b"", "stderr": b""}


def _command_record(argv: list[str], *, cwd: str | None = None) -> dict:
    raw = _run_raw(argv, cwd=cwd)
    record = {"argv": raw["argv"]}
    if "error" in raw:
        record["error"] = raw["error"]
        return record
    record["returncode"] = raw["returncode"]
    record["stdout"] = raw["stdout"].decode("utf-8", errors="replace").strip()[:16_384]
    stderr = raw["stderr"].decode("utf-8", errors="replace").strip()[:4_096]
    if stderr:
        record["stderr"] = stderr
    return record


def _git_snapshot(scope: str) -> dict:
    scope = os.path.abspath(scope)
    top = _command_record(["git", "-C", scope, "rev-parse", "--show-toplevel"])
    if top.get("returncode") != 0 or not top.get("stdout"):
        return {"available": False, "probe": top}
    root = top["stdout"]
    relative_scope = os.path.relpath(scope, root)
    revision = _command_record(["git", "-C", root, "rev-parse", "HEAD"])
    status_raw = _run_raw(
        ["git", "-C", root, "status", "--porcelain=v1", "--untracked-files=all", "--", relative_scope],
        timeout=20.0,
    )
    diff_raw = _run_raw(
        ["git", "-C", root, "diff", "--binary", "HEAD", "--", relative_scope],
        timeout=20.0,
    )
    if ("error" in status_raw or "error" in diff_raw
            or status_raw.get("returncode") != 0 or diff_raw.get("returncode") != 0
            or revision.get("returncode") != 0):
        return {
            "available": False,
            "root": _display_path(root),
            "revision": revision.get("stdout"),
            "status_error": _redact_public(status_raw.get("error")),
            "diff_error": _redact_public(diff_raw.get("error")),
            "status_stderr": _redact_public(
                status_raw.get("stderr", b"").decode("utf-8", errors="replace")[:4_096]
            ),
            "diff_stderr": _redact_public(
                diff_raw.get("stderr", b"").decode("utf-8", errors="replace")[:4_096]
            ),
        }
    status = status_raw["stdout"].decode("utf-8", errors="replace")
    diff = diff_raw["stdout"]
    status_lines = status.splitlines()
    return {
        "available": True,
        "root": _display_path(root),
        "scope": relative_scope,
        "revision": revision.get("stdout"),
        "dirty": bool(status.strip()),
        "status_porcelain": status_lines[:2_000],
        "status_line_count": len(status_lines),
        "status_truncated": len(status_lines) > 2_000,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_diff_bytes": len(diff),
    }


def _sysctl_integer(name: str) -> int | None:
    record = _command_record(["sysctl", "-n", name])
    try:
        return int(record.get("stdout", ""))
    except ValueError:
        return None


def _sysctl_text(name: str) -> str | None:
    record = _command_record(["sysctl", "-n", name])
    value = record.get("stdout")
    return value if isinstance(value, str) and value else None


def _power_source_record() -> dict:
    record = _command_record(["pmset", "-g", "batt"])
    if "stdout" in record:
        record["stdout"] = re.sub(r"\s*\(id=\d+\)", "", record["stdout"])
    return record


def _device_snapshot() -> dict:
    raw = _run_raw(["system_profiler", "SPDisplaysDataType", "-json"])
    if "error" in raw:
        return {"probe_argv": raw["argv"], "error": raw["error"]}
    if raw["returncode"] != 0:
        return {
            "probe_argv": raw["argv"],
            "returncode": raw["returncode"],
            "stderr": raw["stderr"].decode("utf-8", errors="replace")[:4_096],
        }
    try:
        payload = json.loads(raw["stdout"])
    except json.JSONDecodeError as exc:
        return {"probe_argv": raw["argv"], "parse_error": str(exc)}
    allowed = {
        "_name",
        "spdisplays_mtlgpufamilysupport",
        "spdisplays_vendor",
        "sppci_bus",
        "sppci_cores",
        "sppci_device_type",
        "sppci_model",
    }
    return {
        "probe_argv": raw["argv"],
        # Deliberately exclude display serial numbers and the host name. They
        # do not affect kernel performance and should not leak into artifacts.
        "gpus": [
            {key: value for key, value in gpu.items() if key in allowed}
            for gpu in payload.get("SPDisplaysDataType", [])
        ],
    }
