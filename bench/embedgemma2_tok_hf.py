#!/usr/bin/env python3
"""The Hugging Face `tokenizers` lane that `bench_paired` alternates with `bench_embedgemma2_tok`.

    ~/.venvs/ml-2026-10/bin/python bench/embedgemma2_tok_hf.py corpus target/embedgemma2_tok_bench_corpus.bin
    BENCH_TOK_CORPUS=target/embedgemma2_tok_bench_corpus.bin \\
        ~/.venvs/ml-2026-10/bin/python bench/embedgemma2_tok_hf.py

`corpus PATH` writes the fixed corpus both lanes read: records `name TAB text`
separated by NUL. It is seeded, so rerunning rewrites identical bytes. The
workloads:

  short_queries  2000 search-style queries of 5 to 14 words
  documents       200 documents of 120 to 400 words, with digits, punctuation and code lines
  multilingual    500 strings of 4 to 30 words over CJK, Devanagari, Arabic, Thai, Cyrillic, Greek and emoji
  long_100k         1 text of 100000 characters (the BPE sees it as one word)

Without arguments it times `tokenizers.Tokenizer` from the snapshot's
`tokenizer.json` (EMBEDGEMMA2_SNAPSHOT, default the Hugging Face cache) on that
corpus and prints one JSON array like the Rust lane's: `{workload, ms_min,
ms_median, iters, tokens, bytes, ids_fnv32}` per workload and `load`.
`ms_min` is the fastest of BENCH_ITERS (default 5) full passes after one
warm-up pass, `load` the fastest of BENCH_LOAD_ITERS (default 5)
`Tokenizer.from_file` calls. `ids_fnv32` is the FNV-1a hash of every id of a
pass in order, the Rust lane's too: the two must agree before the times mean
anything, and `--check` (or `BENCH_TOK_CHECK=1`) does not time, it only prints
the hashes.

BENCH_TOK_HF_MODE picks what a pass calls:
  loop   `Tokenizer.encode(text)` for each text, one at a time (default): the
         like-for-like lane against `tessl::embedgemma2_tok`, which is single threaded
  batch  `Tokenizer.encode_batch(texts)`, which the library runs on all cores (rayon):
         a reference for what the library gives when it may use the machine
Both take add_special_tokens=True, as `encode_raw(text, true)` does.
"""

import glob
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261016

WORDS = ("metal kernel attention window encoder residual gradient tensor embedding retrieval query document cosine "
         "latency batch pooling matryoshka normalize sequence token position rotary frequency how do speed up "
         "apple silicon train model install error fix best way compare difference between python rust search "
         "result memory cache layer weights float precision benchmark throughput").split()
VERBS = ["speed up", "profile", "debug", "install", "compare", "optimize", "benchmark", "quantize", "fine-tune", "export"]
NOUNS = ["attention kernels", "a tokenizer", "the embedding model", "GEMM", "a Metal shader", "batch norm", "RoPE",
         "a BPE vocabulary", "sentence transformers", "the KV cache", "a rust crate", "mean pooling"]
PLACES = ["Apple silicon", "an M5 Pro", "macOS", "a laptop", "a GPU", "the CPU", "Linux", "a phone"]
CODE = ["fn main() { let v: Vec<u32> = (0..10).collect(); }", "def f(x, y=3): return [i ** 2 for i in range(x)]",
        "SELECT a, COUNT(*) FROM t WHERE b >= 10 GROUP BY a;", "for f in *.rs; do echo \"$f\"; done",
        "{\"a\": [1, 2.5e-3, null], \"b\": {\"c\": \"d\"}}"]
SCRIPTS = ["混合精度训练与推理 東京で会いましょう 日本語のテキストです 你好，世界", "नमस्ते दुनिया हिन्दी भाषा में पाठ क्षत्रिय ज्ञान",
           "مرحبا بالعالم اللغة العربية اختبار النص", "สวัสดีชาวโลก ภาษาไทย ทดสอบ", "Привет мир Γειά σου Κόσμε",
           "😀 🚀 🔥 👍🏽 👨‍👩‍👧‍👦 🇯🇵 ❤️"]


def build_corpus():
    rng = random.Random(SEED)
    out = []
    for _ in range(2000):
        k = rng.randrange(0, 4)
        if k == 0:
            q = f"how do i {rng.choice(VERBS)} {rng.choice(NOUNS)} on {rng.choice(PLACES)}?"
        elif k == 1:
            q = " ".join(rng.choice(WORDS) for _ in range(rng.randrange(5, 15)))
        elif k == 2:
            q = f"{rng.choice(NOUNS)} vs {rng.choice(NOUNS)} {rng.randrange(1, 100)}x {rng.choice(WORDS)}"
        else:
            q = f"{rng.choice(WORDS).capitalize()} {rng.choice(WORDS)}-{rng.choice(WORDS)} ({rng.randrange(2000, 2030)})"
        out.append(("short_queries", q))
    for _ in range(200):
        sents = []
        for _ in range(rng.randrange(8, 24)):
            n = rng.randrange(8, 22)
            s = " ".join(rng.choice(WORDS) + (str(rng.randrange(0, 1000)) if rng.randrange(0, 6) == 0 else "") for _ in range(n))
            sents.append(s.capitalize() + rng.choice([".", ".", ".", "!", "?", ";"]))
            if rng.randrange(0, 8) == 0:
                sents.append("\n    " + rng.choice(CODE) + "\n")
        out.append(("documents", " ".join(sents)))
    for _ in range(500):
        out.append(("multilingual", " ".join(rng.choice(SCRIPTS).split()[rng.randrange(0, 3)] if rng.randrange(0, 3) else rng.choice(WORDS)
                                              for _ in range(rng.randrange(4, 31)))))
    sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
    import gen_embedgemma2_tok_fixtures as gen
    out.append(("long_100k", gen.long_text(100_000)))
    return out


def write_corpus(path):
    recs = build_corpus()
    with open(path, "wb") as f:
        f.write(b"\0".join(f"{n}\t{t}".encode("utf-8") for n, t in recs))
    print(f"{len(recs)} records, {os.path.getsize(path)} bytes -> {path}")


def snapshot():
    if os.environ.get("EMBEDGEMMA2_SNAPSHOT"):
        return os.environ["EMBEDGEMMA2_SNAPSHOT"]
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/*/config.json")))
    if not hits:
        raise SystemExit("no google/embeddinggemma-2 snapshot; set EMBEDGEMMA2_SNAPSHOT")
    return os.path.dirname(hits[-1])


def fnv32(h, ids):
    for i in ids:
        for b in i.to_bytes(4, "little"):
            h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def read_corpus(path):
    workloads = {}
    for rec in open(path, "rb").read().split(b"\0"):
        if not rec:
            continue
        name, _, text = rec.decode("utf-8").partition("\t")
        workloads.setdefault(name, []).append(text)
    return workloads


def count(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    if not v.isdigit() or int(v) < 1:
        raise SystemExit(f"{name}={v!r}: expected an integer of at least 1")
    return int(v)


def median(v):
    return sorted(v)[len(v) // 2]


def main():
    from tokenizers import Tokenizer

    path = os.environ.get("BENCH_TOK_CORPUS")
    if not path:
        raise SystemExit("BENCH_TOK_CORPUS must name the corpus (see the module docstring)")
    check = "--check" in sys.argv or os.environ.get("BENCH_TOK_CHECK") == "1"
    mode = os.environ.get("BENCH_TOK_HF_MODE", "loop")
    if mode not in ("loop", "batch"):
        raise SystemExit(f"BENCH_TOK_HF_MODE={mode!r}: expected loop or batch")
    iters, load_iters = count("BENCH_ITERS", 5), count("BENCH_LOAD_ITERS", 5)
    json_path = os.path.join(snapshot(), "tokenizer.json")
    workloads = read_corpus(path)

    loads = []
    tok = None
    for _ in range(load_iters):
        t0 = time.perf_counter()
        tok = Tokenizer.from_file(json_path)
        loads.append((time.perf_counter() - t0) * 1e3)
    rows = [{"workload": "load", "ms_min": min(loads), "ms_median": median(loads), "iters": load_iters,
             "tokens": tok.get_vocab_size(), "bytes": os.path.getsize(json_path), "ids_fnv32": 0}]

    for name, texts in workloads.items():
        def one_pass():
            if mode == "batch":
                return [e.ids for e in tok.encode_batch(texts)]
            return [tok.encode(t).ids for t in texts]

        h, tokens = 2166136261, 0
        for ids in one_pass():
            tokens += len(ids)
            h = fnv32(h, ids)
        times = []
        if not check:
            for _ in range(iters):
                t0 = time.perf_counter()
                one_pass()
                times.append((time.perf_counter() - t0) * 1e3)
        rows.append({"workload": name, "ms_min": min(times) if times else 0.0, "ms_median": median(times) if times else 0.0,
                     "iters": iters, "tokens": tokens, "bytes": sum(len(t.encode()) for t in texts), "ids_fnv32": h})

    for r in rows:
        rate = "" if r["workload"] == "load" or not r["ms_min"] else f"  {r['tokens'] / (r['ms_min'] / 1e3):.0f} tokens/s"
        print(f"{r['workload']:<14} min {r['ms_min']:>10.3f} ms  median {r['ms_median']:>10.3f} ms  "
              f"{r['tokens']} tokens, {r['bytes']} bytes{rate}  [{mode}]", file=sys.stderr)
    print(json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "corpus":
        write_corpus(sys.argv[2])
    else:
        main()
