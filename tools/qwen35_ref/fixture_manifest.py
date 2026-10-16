"""The one manifest format for tessl's generator-backed fixture directories.

`tests/fixtures/qwen35/`, `qwen35_train/` and `qwen35_train_grouped/` each hold a
`MANIFEST.json` written by this module and checked by `tests/fixture_manifests.rs`
(which lists every file present, so a file nobody accounted for fails the test).

    {
      "schema": 1,
      "generator": "scripts/gen_qwen35_fixtures.py",
      "generator_commit": "<40 hex>",         # last commit touching the generator
      "versions": {"torch": ..., "transformers": ..., "numpy": ..., "python": ...},
      "versions_role": "produced" | "reproduced",
      "inputs_source": "<where the inputs come from, and how stable that is>",
      "files": {"<name>": {"role": "input|golden|model|config", "published": bool, "note": str}},
      "reproduction": {...}                    # only when versions_role is "reproduced"
    }

`versions_role` says what the versions mean. "produced": this run wrote the files.
"reproduced": the committed files predate the manifest and their original build is
unrecorded; the versions are those of a regeneration into a scratch directory that
was compared with them, and `reproduction` carries the comparison.

`published` is a per-file status, not a quality mark. True: the file is a golden of
record (or an input to one) and new tests may hold a kernel to it. False: it is kept
for the tests that already read it, and nothing new may use it as a golden.
"""

import json
import os
import platform
import subprocess

import numpy as np

SCHEMA = 1
ROOTS = ("input", "golden", "model", "config")


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def generator_commit(generator):
    """The last commit that touched `generator`; refuses a dirty generator, because
    a manifest naming a commit the file no longer matches would be a false record."""
    root = repo_root()
    dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", generator],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{generator} has uncommitted changes; commit it first so the manifest can name it")
    sha = subprocess.run(["git", "-C", root, "log", "-1", "--format=%H", "--", generator],
                         capture_output=True, text=True, check=True).stdout.strip()
    if len(sha) != 40:
        raise SystemExit(f"{generator} has no commit yet; commit it first")
    return sha


def versions():
    import torch
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "python": platform.python_version()}


def compare(committed_dir, scratch_dir):
    """`{file: max |a-b| / max|a|}` over every differing .npy both directories hold,
    plus the byte-identical names and the non-npy files that differ."""
    moved, identical, differs = {}, [], []
    for name in sorted(os.listdir(committed_dir)):
        a_path, b_path = os.path.join(committed_dir, name), os.path.join(scratch_dir, name)
        if name == "MANIFEST.json" or not os.path.isfile(b_path):
            continue
        with open(a_path, "rb") as fa, open(b_path, "rb") as fb:
            if fa.read() == fb.read():
                identical.append(name)
                continue
        if not name.endswith(".npy"):
            differs.append(name)
            continue
        if name.endswith(".npy"):
            a, b = np.load(a_path).astype(np.float64), np.load(b_path).astype(np.float64)
            peak = float(np.abs(a).max()) or 1.0
            moved[name] = float(np.abs(a - b).max()) / peak
    return {"byte_identical": identical, "differing_non_npy": differs, "max_relative_to_peak": moved}


def write_manifest(out_dir, generator, inputs_source, files, role, reproduction=None, extra=None):
    listed = set(files)
    present = {n for n in os.listdir(out_dir) if n != "MANIFEST.json"}
    if listed != present:
        raise SystemExit(f"manifest and directory disagree: unlisted {sorted(present - listed)}, "
                         f"missing {sorted(listed - present)}")
    for name, meta in files.items():
        if meta["role"] not in ROOTS or not isinstance(meta["published"], bool):
            raise SystemExit(f"{name}: bad role/published {meta}")
    manifest = {
        "schema": SCHEMA,
        "generator": generator,
        "generator_commit": generator_commit(generator),
        "versions": versions(),
        "versions_role": role,
        "inputs_source": inputs_source,
        "files": {n: files[n] for n in sorted(files)},
    }
    if reproduction is not None:
        manifest["reproduction"] = reproduction
    if extra:
        manifest.update(extra)
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
