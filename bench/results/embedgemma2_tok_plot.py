"""Plot for bench/results/embedgemma2_tok_m5pro.txt (analysis only; ships nothing).

Reads embedgemma2_tok_m5pro.json (`bench_paired` over `bench_embedgemma2_tok`
and bench/embedgemma2_tok_hf.py, loop and batch) and the two lanes' own
output for the token counts and id hashes (embedgemma2_tok_m5pro_tessl_lane.json,
embedgemma2_tok_m5pro_hf_lane.json), and writes embedgemma2_tok_m5pro.png:
left, encode throughput (tokens per second at the fastest of the interleaved
rounds, each round itself the fastest of its passes; every round is a dot);
right, the load time of the 32 MB tokenizer.json the same way. It refuses to
plot if the two lanes' id hashes differ: the lanes must have produced the same
ids before the times mean anything.

    uv run --no-project --with matplotlib python embedgemma2_tok_plot.py <results dir>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
doc = json.loads((root / "embedgemma2_tok_m5pro.json").read_text())
rust = {r["workload"]: r for r in json.loads((root / "embedgemma2_tok_m5pro_tessl_lane.json").read_text())}
hf = {r["workload"]: r for r in json.loads((root / "embedgemma2_tok_m5pro_hf_lane.json").read_text())}
summary = {row["workload"]: row for row in doc["summary"]}

WORKLOADS = [w for w in summary if w != "load"]
for w in WORKLOADS:
    if rust[w]["ids_fnv32"] != hf[w]["ids_fnv32"] or rust[w]["tokens"] != hf[w]["tokens"]:
        sys.exit(f"{w}: the lanes produced different ids ({rust[w]['ids_fnv32']} vs {hf[w]['ids_fnv32']})")

LANES = [("tessl", "tessl (Rust, 1 thread)", "#4c72b0"),
         ("hf_loop", "tokenizers encode() loop (1 thread)", "#dd8452"),
         ("hf_batch", "tokenizers encode_batch() (all cores)", "#8c8c8c")]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.4, 1]})

width = 0.26
for j, (lane, label, color) in enumerate(LANES):
    for i, w in enumerate(WORKLOADS):
        rounds = summary[w][lane]["ms_min_all"]
        tokens = rust[w]["tokens"]
        rate = [tokens / (ms / 1e3) for ms in rounds]
        x = i + (j - 1) * width
        ax.bar(x, max(rate), width * 0.9, color=color, alpha=0.85, label=label if i == 0 else None)
        ax.plot([x] * len(rate), rate, "o", ms=3, color="black", alpha=0.45)
        ax.text(x, max(rate) * 1.08, f"{max(rate) / 1e6:.2f}M", ha="center", fontsize=7)
ax.set_yscale("log")
ax.set_xticks(range(len(WORKLOADS)), [f"{w}\n{rust[w]['tokens']} tokens" for w in WORKLOADS])
ax.set_ylabel("tokens per second (log; bar: best round, dots: every round)")
ax.set_title("Encode throughput, interleaved rounds (identical ids on every workload)")
ax.legend(loc="lower right", fontsize=8)

for j, (lane, label, color) in enumerate(LANES):
    rounds = summary["load"][lane]["ms_min_all"]
    bx.bar(j, min(rounds), 0.7, color=color, alpha=0.85)
    bx.plot([j] * len(rounds), rounds, "o", ms=3, color="black", alpha=0.45)
    bx.text(j, min(rounds) * 1.05, f"{min(rounds):.0f} ms", ha="center", fontsize=8)
bx.set_xticks(range(len(LANES)), ["tessl", "tokenizers\n(loop lane)", "tokenizers\n(batch lane)"], fontsize=8)
bx.set_ylabel("ms (best round; dots: every round)")
bx.set_title("Load tokenizer.json (32 MB, 262144 tokens)")

fig.suptitle(f"embedgemma2_tok vs Hugging Face tokenizers 0.23.2, {doc['rounds']} interleaved rounds, loaded M5 Pro", fontsize=10)
fig.tight_layout()
fig.savefig(root / "embedgemma2_tok_m5pro.png", dpi=140)
print("wrote", root / "embedgemma2_tok_m5pro.png")
