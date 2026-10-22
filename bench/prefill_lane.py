#!/usr/bin/env python3
"""The shared half of the Qwen3.5-2B prefill+score head-to-head lanes.

Three lanes run the same work on the same ids and print the same JSON, so
`bench_paired` can interleave them (card ft-05f5b7709a71a3774120e617a5342816, AC1):

  tessl   target/release/bench_qwen35_layers --prefill-score   (Rust, src/bin)
  torch   bench/qwen35_prefill_torch.py                        (torch 2.14.1 MPS, transformers 5.19.0)
  mlx     bench/qwen35_prefill_mlx.py                          (mlx-lm 0.32.0, mlx 0.32.3)

THE WORK (the same definition the 2026-10-04 PyTorch 2.12.1 rerun used, ojas
bench/results/2026-10-03-lappi-inference-rerun/research-notes.md): the first T
ids of the ids file through the embedding gather, all 24 layers with the
recurrent and KV state kept, and the final norm; then the 17 answer tokens'
logits at the LAST position only, as the f32 dot of the f32 normed hidden row
with the answers' embedding rows (tied head), and their log-softmax over the
17. No full-vocabulary head is formed on any side. The timed span runs from
the host ids to the 17 log-probabilities on the host.

THE IDS: `tests/fixtures/qwen35_prefill_ids.json`, one file read by all three
lanes (this module's `load_ids` / `prefix`; Rust's `src/bin/common/prefill_ids.rs`).
A lane at length T runs `ids[:T]`. The file carries the FNV-1a 32 digest of the
prefix for each length it was built for; a length it has none for, a digest that
disagrees, a token outside the vocabulary, a file shorter than T, or a lane whose
forward ran on a different number of tokens than asked is an error, never a
shorter or different run. Regenerate it (tokenizes docs/*.md with the snapshot's
tokenizer, so the tokens are real text):

  ~/.venvs/ml-2026-10/bin/python bench/prefill_lane.py --generate

OUTPUT: stdout's last line is the JSON array `bench_paired --format workloads`
reads. Per length T two rows: `prefill_score_t{T}` (`ms_min` = the fastest timed
round in that process) and `prefill_score_t{T}_median` (`ms_min` = the median
timed round, so the report carries per-lane min and median). Both rows carry
`tokens` (the length that really ran), `answers`, `ids_fnv1a32`, `ms_median`,
`iters`. Lengths alternate inside the process, the order reversing every other
round. The first round's answer log-probabilities go to stderr as
`SCORES t=.. logprobs=[..]` to compare the three sides' numerics.

KNOBS (same names on every lane): BENCH_ITERS (timed rounds, default 9) and
BENCH_WARMUP (untimed rounds, default 3), or --iters / --warmup.

COMMANDS (exact; run from the repo root on a quiet machine, under the machine
lock, with every lane and bench_paired built first):

  cargo build --release --bin bench_qwen35_layers --bin bench_paired
  SNAP=/Users/bharath/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
  PY=/Users/bharath/.venvs/ml-2026-10/bin/python
  IDS=tests/fixtures/qwen35_prefill_ids.json
  target/release/bench_paired --evidence --rounds 6 --clean-env \\
    --out bench/results/qwen35_prefill_score_h2h.json \\
    "tessl:BENCH_ITERS=9,BENCH_WARMUP=3=target/release/bench_qwen35_layers --prefill-score --weights=$SNAP --ids=$IDS 200 1024 2048 8192" \\
    "torch:BENCH_ITERS=9,BENCH_WARMUP=3=$PY bench/qwen35_prefill_torch.py --snapshot $SNAP --ids $IDS --lengths 200,1024,2048,8192" \\
    "mlx:BENCH_ITERS=9,BENCH_WARMUP=3=$PY bench/qwen35_prefill_mlx.py --snapshot $SNAP --ids $IDS --lengths 200,1024,2048,8192"

(`--format workloads` is the default; the first lane is the base, so the table
reads torch/tessl and mlx/tessl, and every row where a lane's ratio to tessl
is below 1.0 is a lane where tessl loses and must be reported. Rounds: an even
count of at least 4; each round runs every lane once, in order on even rounds
and reversed on odd ones.)

torch 2.15: NOT RUN. It is not installed in ~/.venvs/ml-2026-10 (torch 2.14.1)
and installing it is a dependency decision for the owner. When it exists, add a
fourth lane line (`torch215:...=$PY215 bench/qwen35_prefill_torch.py ...`)
on the same ids file; its column is absent from any report until then, not
silently omitted.

Torch lane details: bf16 (the checkpoint's dtype, as the earlier run) via
transformers' default sdpa attention; transformers' torch fallback for the GDN
layers ("fast path is not available": no `fla` on MPS), as in the earlier run.
MLX lane details: the checkpoint's bf16 weights, not quantized; mlx-lm's
`qwen3_5` model run in one pass (no chunked prefill) with a fresh cache per call.
"""
import argparse
import json
import os
import statistics
import sys
import time

SCHEMA = "tessl-prefill-ids-v1"
LENGTHS = (200, 1024, 2048, 8192)
ANSWER_LETTERS = "ABCDEFGHIJKLMNOPQ"  # 17 answers, as the earlier run
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IDS = os.path.join(os.path.dirname(HERE), "tests", "fixtures", "qwen35_prefill_ids.json")
SNAPSHOT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/*/config.json")
DEFAULT_ITERS = 9
DEFAULT_WARMUP = 3
MAX_ITERS = 10_000


def fnv1a32(ids):
    """FNV-1a 32 over the ids as little-endian u32 words (Rust's `fnv1a32`)."""
    h = 0x811C9DC5
    for i in ids:
        for b in int(i).to_bytes(4, "little"):
            h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def load_ids(path):
    """The ids file, validated: schema, vocab, every token below it, answers."""
    with open(path) as f:
        doc = json.load(f)
    if doc.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: schema {doc.get('schema')!r}, expected {SCHEMA!r}")
    vocab = doc.get("vocab")
    if not isinstance(vocab, int) or vocab <= 0:
        raise SystemExit(f"{path}: vocab {vocab!r} is not a positive integer")
    for key in ("ids", "answers"):
        v = doc.get(key)
        if not isinstance(v, list) or not v or not all(isinstance(i, int) and not isinstance(i, bool) for i in v):
            raise SystemExit(f"{path}: {key!r} is not a non-empty list of integers")
    bad = [i for i in doc["ids"] + doc["answers"] if not 0 <= i < vocab]
    if bad:
        raise SystemExit(f"{path}: token {bad[0]} outside 0..{vocab}")
    if not isinstance(doc.get("digests"), dict):
        raise SystemExit(f"{path}: no digests object")
    return doc


def prefix(doc, t):
    """The first `t` ids; refuses a short file, a length without a digest, a digest mismatch."""
    ids = doc["ids"]
    if t <= 0 or t > len(ids):
        raise SystemExit(f"length {t} asked of an ids file with {len(ids)} ids")
    want = doc["digests"].get(str(t))
    if want is None:
        raise SystemExit(f"the ids file lists no digest for length {t}; regenerate it with that length")
    got = fnv1a32(ids[:t])
    if got != want:
        raise SystemExit(f"length {t}: ids digest {got} but the file says {want}")
    return ids[:t]


def parse_lengths(text):
    try:
        ts = [int(x) for x in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"--lengths wants comma-separated integers, got {text!r}")
    if not ts or any(t <= 0 for t in ts) or len(set(ts)) != len(ts):
        raise argparse.ArgumentTypeError(f"--lengths must be distinct positive integers, got {text!r}")
    return ts


def env_count(name, default, minimum):
    raw = os.environ.get(name)
    if raw is None:
        return default
    if not raw.isascii() or not raw.isdigit() or (raw != "0" and raw.startswith("0")):
        raise SystemExit(f"{name}={raw!r} is not a non-negative integer")
    v = int(raw)
    if not minimum <= v <= MAX_ITERS:
        raise SystemExit(f"{name}={v} outside {minimum}..{MAX_ITERS}")
    return v


def snapshot_default():
    import glob
    if os.environ.get("QWEN35_SNAPSHOT"):
        return os.environ["QWEN35_SNAPSHOT"]
    hits = sorted(glob.glob(SNAPSHOT_GLOB))
    if not hits:
        raise SystemExit("no Qwen3.5-2B-Base snapshot; pass --snapshot or set QWEN35_SNAPSHOT")
    return os.path.dirname(hits[-1])


def base_parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--snapshot", default=None, help="Qwen3.5-2B-Base snapshot directory (default: the local HF cache's)")
    ap.add_argument("--ids", default=DEFAULT_IDS, help="the ids file every lane reads")
    ap.add_argument("--lengths", type=parse_lengths, default=list(LENGTHS))
    ap.add_argument("--iters", type=int, default=None, help="timed rounds (default BENCH_ITERS or 9)")
    ap.add_argument("--warmup", type=int, default=None, help="untimed rounds (default BENCH_WARMUP or 3)")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                    help="cpu is for the device-free smoke test only; its rows are labelled")
    return ap


def resolve_counts(args):
    iters = args.iters if args.iters is not None else env_count("BENCH_ITERS", DEFAULT_ITERS, 1)
    warmup = args.warmup if args.warmup is not None else env_count("BENCH_WARMUP", DEFAULT_WARMUP, 0)
    if not 1 <= iters <= MAX_ITERS or not 0 <= warmup <= MAX_ITERS:
        raise SystemExit(f"iters {iters} / warmup {warmup} out of range")
    return iters, warmup


def row(name, backend, ms, iters, ms_median, tokens, answers, digest):
    return {"workload": name, "backend": backend, "ms_min": round(ms, 4), "ms_median": round(ms_median, 4),
            "iters": iters, "tokens": tokens, "answers": answers, "ids_fnv1a32": digest}


def format_rows(backend, lengths, samples, iters, answers, digests):
    """Two rows per length, the shape `bench_paired --format workloads` reads."""
    rows = []
    for t in lengths:
        ms = samples[t]
        if len(ms) != iters:
            raise SystemExit(f"T={t}: {len(ms)} samples for {iters} iterations")
        med = statistics.median(ms)
        rows.append(row(f"prefill_score_t{t}", backend, min(ms), iters, med, t, answers, digests[t]))
        rows.append(row(f"prefill_score_t{t}_median", backend, med, iters, med, t, answers, digests[t]))
    return rows


def run_rounds(lengths, iters, warmup, one, first_round_sink=None):
    """Alternate the lengths round by round (reversed on odd rounds).

    `one(t)` runs one timed sample at length t and returns (ms, logprobs,
    tokens_run). A forward that ran on other than t tokens, or that returned a
    non-finite or constant-zero score row, aborts the lane.
    """
    import math
    samples = {t: [] for t in lengths}
    for rnd in range(warmup + iters):
        order = list(lengths) if rnd % 2 == 0 else list(reversed(lengths))
        for t in order:
            ms, logprobs, ran = one(t)
            if ran != t:
                raise SystemExit(f"T={t}: the forward ran on {ran} tokens")
            if not logprobs or not all(math.isfinite(x) for x in logprobs) or all(x == 0 for x in logprobs):
                raise SystemExit(f"T={t}: answer log-probabilities are not finite and non-zero: {logprobs}")
            if rnd == 0 and first_round_sink is not None:
                first_round_sink(t, logprobs)
            if rnd >= warmup:
                samples[t].append(ms)
    return samples


def emit(rows):
    print(json.dumps(rows))


def sink_scores(t, logprobs):
    print(f"SCORES t={t} logprobs={[round(x, 5) for x in logprobs]}", file=sys.stderr, flush=True)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1e3, out


def prepare(args):
    """Ids, digests and answers for the requested lengths, all validated."""
    doc = load_ids(args.ids)
    ids = {t: prefix(doc, t) for t in args.lengths}
    return doc, ids, {t: fnv1a32(v) for t, v in ids.items()}


def generate(argv):
    """Write the ids file: the first 8192 tokens of docs/*.md, and the answer letters."""
    import glob
    from tokenizers import Tokenizer
    ap = argparse.ArgumentParser(prog="prefill_lane.py --generate")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--out", default=DEFAULT_IDS)
    ap.add_argument("--docs", default=os.path.join(os.path.dirname(HERE), "docs", "*.md"))
    args = ap.parse_args(argv)
    snap = args.snapshot or snapshot_default()
    tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
    files = sorted(glob.glob(args.docs))
    text = "\n\n".join(open(p, encoding="utf-8").read() for p in files)
    ids = tok.encode(text, add_special_tokens=False).ids
    need = max(LENGTHS)
    if len(ids) < need:
        raise SystemExit(f"{len(files)} docs give {len(ids)} tokens, need {need}")
    ids = ids[:need]
    answers = []
    for ch in ANSWER_LETTERS:
        enc = tok.encode(ch, add_special_tokens=False).ids
        if len(enc) != 1:
            raise SystemExit(f"answer {ch!r} is {len(enc)} tokens")
        answers.append(enc[0])
    if len(set(answers)) != len(answers):
        raise SystemExit("answer tokens are not distinct")
    with open(os.path.join(snap, "config.json")) as f:
        vocab = json.load(f)["text_config"]["vocab_size"]
    doc = {"schema": SCHEMA,
           "snapshot": os.path.basename(snap.rstrip("/")),
           "source": "first %d tokens of the tokenized concatenation of %s, sorted by name (%d files); "
                     "answers are the tokens of the letters A..Q" % (need, "docs/*.md", len(files)),
           "vocab": vocab,
           "digests": {str(t): fnv1a32(ids[:t]) for t in LENGTHS},
           "answers": answers,
           "ids": ids}
    with open(args.out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {args.out}: {len(ids)} ids, {len(answers)} answers, digests {doc['digests']}")


if __name__ == "__main__":
    if "--generate" in sys.argv[1:]:
        generate(sys.argv[1:])
    else:
        raise SystemExit("prefill_lane.py is a library; run with --generate to write the ids file")
