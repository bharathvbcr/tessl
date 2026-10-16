#!/usr/bin/env python3
"""Generate the (text, ids) oracles for src/embedgemma2_tok.rs, from the Hugging Face tokenizers.

    ~/.venvs/ml-2026-10/bin/python scripts/gen_embedgemma2_tok_fixtures.py tiny
    ~/.venvs/ml-2026-10/bin/python scripts/gen_embedgemma2_tok_fixtures.py real [SNAPSHOT_DIR]

Needs numpy and `tokenizers`; `real` also needs torch, transformers >= 5.19 and
sentence-transformers >= 6.1 (the versions the e2e references were taken with).
Deterministic: the corpus comes from fixed strings and `random.Random(SEED)`,
and the oracles are pure functions of the tokenizer files, so rerunning
rewrites identical files.

`tiny` reads the three hand-written tokenizers in tests/fixtures/embedgemma2_tok/
(tiny_gemma.json: the snapshot's shape, with byte fallback and added tokens;
tiny_legacy.json: a real space split, `"a b"` merges, ignore_merges, fuse_unk,
a repeated merge; tiny_bytes.json: a byte-fallback table with gaps, so an
unknown-character run meets byte tokens) and writes, next to each, a directory
of the same name without `.json` (the tiny fixtures are small and committed).
The oracle is the library alone: `Tokenizer.encode(text, add_special_tokens=...)`
and `Tokenizer.decode(ids, skip_special_tokens=...)`.

`real` writes into $EMBEDGEMMA2_TOK_REF_DIR (default target/embedgemma2_tok_ref/,
not committed: the corpus is several MB). The oracle is the path
sentence-transformers feeds the model: `SentenceTransformer.preprocess([text],
prompt=PROMPT)` on the snapshot, one text at a time (a batch of one: see the
batch quirk below). Each case is also run through the library alone with the one
rule that path adds (no `<bos>`/`<eos>` when the rendered text starts with
`<bos>`), and the two must agree, so the oracle is checked against itself. A
text that holds `<|image|>`, `<|video|>` or `<|audio|>` makes the processor
raise; those cases (status 1) carry the library's ids instead, which
`Tokenizer::encode_raw` is held to.

Batch quirk, asserted here so a transformers fix is noticed: the processor
looks at the FIRST text of a batch only to decide whether to add `<bos>`/`<eos>`
for the whole batch (`ProcessorMixin.apply_chat_template`: `single_prompt =
prompt[0]`), so ["<bos>hi", "plain"] gives "plain" no `<bos>` and no `<eos>`.
The Rust tokenizer applies the rule to each text on its own.

Corpus files (every array is int64 `.npy`; an empty list is stored as [0] and
the offsets say how long each really is):
  texts.bin, text_offsets   the texts, concatenated UTF-8, and N+1 byte offsets
  prompts.txt, prompt_idx   prompt names one per line; per case, a line index or -1
  groups.txt, group_idx     corpus names one per line; per case, a line index
  status                    0 ok; 1 the processor refuses the text
  st_ids, st_ids_offsets    the ids the model receives (status 1: the library's, with the rule)
  raw_ids, raw_ids_offsets  `Tokenizer.encode(rendered, add_special_tokens=True)`
  plain_ids, plain_ids_offsetsthe same with add_special_tokens=False
  dec_ids, dec_id_offsets, dec_skip   decode inputs and skip_special_tokens
  dec_out.bin, dec_out_offsets        the library's decoded strings
  dec_groups.txt, dec_group_idx       decode corpus names
`rendered` is the prompt text followed by the text.
"""

import importlib.metadata
import json
import os
import random
import sys

import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
TINY_DIR = os.path.join(ROOT, "tests", "fixtures", "embedgemma2_tok")
SEED = 20261016
BOS = "<bos>"


def save_i64(path, values):
    arr = np.array(list(values) if len(values) else [0], dtype=np.int64)
    np.save(path, arr)


def write_blob(out, name, strings):
    data = [s.encode("utf-8") for s in strings]
    offsets = [0]
    for d in data:
        offsets.append(offsets[-1] + len(d))
    with open(os.path.join(out, name + ".bin"), "wb") as f:
        f.write(b"".join(data))
    return offsets


def write_ragged(out, name, rows):
    flat, offsets = [], [0]
    for r in rows:
        flat.extend(r)
        offsets.append(len(flat))
    save_i64(os.path.join(out, name + ".npy"), flat)
    save_i64(os.path.join(out, name + "_offsets.npy"), offsets)


def write_lines(out, name, lines):
    for line in lines:
        assert "\n" not in line
    with open(os.path.join(out, name), "w", encoding="utf-8") as f:
        f.write("".join(line + "\n" for line in lines))


def index_of(names):
    order = sorted(set(names))
    return order, [order.index(n) for n in names]


# ---------------------------------------------------------------- the corpus

# Pools for the random fuzz set: ordinary text of several scripts, combining
# marks, controls, private use, characters outside the vocabulary, and the
# pieces of the special tokens.
LATIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
PUNCT = ".,;:!?'\"()[]{}<>|/\\-_+=*&^%$#@~`"
SPACES = [" ", " ", " ", "  ", "\t", "\n", "\r\n", "\u00a0", "\u3000", "\u2003", "\u200b"]
CJK = "混合精度训练与推理東京で会いましょう日本語한국어漢字测试的是不了人我有"
DEVANAGARI = "नमस्ते दुनिया हिन्दी भाषा क्षत्रिय ज्ञान"
ARABIC = "مرحبا بالعالم اللغة العربية اختبار"
THAI = "สวัสดีชาวโลก ภาษาไทย ทดสอบ"
OTHER = "Привет мир Γειά σου Κόσμε שלום עולם àéîõü ñß ç"
COMBINING = ["\u0301", "\u0308", "\u0323", "\u200d", "\u20e3", "\ufe0f", "\u0e31", "\u093c"]
EMOJI = ["😀", "🚀", "🔥", "👍🏽", "👨\u200d👩\u200d👧\u200d👦", "🏳️\u200d🌈", "🇯🇵", "🇺🇸", "1️⃣", "❤️", "🧑🏿\u200d💻"]
CONTROL = [chr(c) for c in list(range(0, 32)) + [127] + list(range(128, 160))]
PRIVATE = ["\ue000", "\uf8ff", "\U000f0000", "\U0010ffff", "\U0010fffd"]
SPECIAL_PIECES = ["<bos>", "<eos>", "<pad>", "<unk>", "<mask>", "<|turn>", "<turn|>", "<|tool>", "<tool|>",
                  "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", '<|"|>', "<|think|>",
                  "<|channel>", "<channel|>", "<|image>", "<image|>", "<|audio>", "<audio|>",
                  "<bo", "<bos", "bos>", "<|turn", "turn|>", "<", ">", "<|", "|>", "<<bos>>", "<BOS>", "<unused0>",
                  "<0x41>", "<0xE6>"]
# Characters that probably have no vocabulary entry (rare planes and
# unassigned code points); `real` keeps the ones the vocabulary really lacks.
RARE = [chr(c) for c in [0x2fa1d, 0x20000, 0x2a6df, 0x1f9ff, 0x10330, 0x13000, 0x1d400, 0x16a0, 0xa4d0, 0x1e900,
                         0x10fff, 0xfffd, 0xfffe, 0xffff, 0xe0001, 0xe0100, 0x1f1e6, 0x2ff0, 0x4dc0, 0x12000]]


def fuzz_string(rng):
    n = rng.randrange(0, 60)
    parts = []
    for _ in range(n):
        k = rng.randrange(0, 15)
        if k == 0:
            parts.append(rng.choice(LATIN))
        elif k == 1:
            parts.append("".join(rng.choice(LATIN) for _ in range(rng.randrange(2, 9))))
        elif k == 2:
            parts.append("".join(rng.choice(DIGITS) for _ in range(rng.randrange(1, 12))))
        elif k == 3:
            parts.append(rng.choice(PUNCT))
        elif k == 4:
            parts.append(rng.choice(SPACES))
        elif k == 5:
            parts.append("".join(rng.choice(CJK) for _ in range(rng.randrange(1, 6))))
        elif k == 6:
            parts.append(rng.choice(DEVANAGARI.split()))
        elif k == 7:
            parts.append(rng.choice(ARABIC.split()))
        elif k == 8:
            parts.append(rng.choice(THAI.split()))
        elif k == 9:
            parts.append(rng.choice(OTHER.split()))
        elif k == 10:
            parts.append(rng.choice(EMOJI))
        elif k == 11:
            parts.append(rng.choice(COMBINING))
        elif k == 12:
            parts.append(rng.choice(CONTROL + PRIVATE + RARE))
        elif k == 13:
            parts.append(rng.choice(SPECIAL_PIECES))
        else:
            parts.append(" ".join(rng.choice(["the", "of", "and", "token", "embedding", "query", "return", "def", "fn"])
                                  for _ in range(rng.randrange(1, 5))))
    return "".join(parts)


def long_text(n_chars):
    words = ("metal kernel attention window encoder residual gradient tensor embedding retrieval query document "
             "cosine latency batch pooling matryoshka normalize sequence token position rotary frequency").split()
    rng = random.Random(SEED + 1)
    pieces, total = [], 0
    while total < n_chars:
        k = rng.randrange(0, 40)
        if k < 28:
            p = rng.choice(words) + str(rng.randrange(0, 1000))
        elif k < 32:
            p = rng.choice(CJK.split("日")) + rng.choice(OTHER.split())
        elif k < 35:
            p = rng.choice(DEVANAGARI.split()) + " " + rng.choice(ARABIC.split())
        elif k < 37:
            p = rng.choice(EMOJI)
        elif k < 38:
            p = "\n\n"
        elif k < 39:
            p = rng.choice(PUNCT) * rng.randrange(1, 5)
        else:
            p = "<bos>" if rng.randrange(0, 4) == 0 else "  " * rng.randrange(1, 4)
        pieces.append(p)
        total += len(p) + 1
    return " ".join(pieces)[:n_chars]


CODE = {
    "python": "def f(x, y=3):\n    \"\"\"Add.\"\"\"\n    return [i ** 2 for i in range(x) if i % y == 0]\n\nclass A:\n\tpass\n",
    "rust": "fn main() {\n    let v: Vec<u32> = (0..10).collect();\n    println!(\"{:?}\", v);\n}\n",
    "json": '{"a": [1, 2.5e-3, null, true], "b": {"c": "d\\n\\u00e9"}}',
    "html": "<div class=\"x\"><p>Hello &amp; welcome</p><br/></div>",
    "sql": "SELECT a, COUNT(*) FROM t WHERE b >= 10 AND c LIKE '%x%' GROUP BY a ORDER BY 2 DESC;",
    "shell": "for f in *.rs; do echo \"$f\" | wc -l; done  # count lines",
}


def build_cases(prompts, absent_chars):
    """[(group, prompt_name or None, text)]."""
    cases = []

    def add(group, text, prompt=None):
        cases.append((group, prompt, text))

    samples = ["How do I speed up attention kernels on Apple silicon?", "", "x",
               "Bidirectional sliding-window attention lets each token see 512 neighbours.",
               "混合精度训练 — 東京 🚀 café naïve résumé", "  leading and trailing  "]
    for name in sorted(prompts):
        for s in samples:
            add("prompts", s, name)
    add("prompts", "<bos>already has one", "query")
    for s in ["", "x", "hello world", "   ", "\t", "\n", "a"]:
        add("plain", s)

    for s in ["", " ", "  ", "   ", " " * 31, " " * 32, " " * 33, " " * 200, "\t", "\t\t\t", "\n", "\n\n", "\n" * 31,
              "\n" * 40, "\r", "\r\n", "\r\n\r\n", "a\r\nb", " a", "a ", " a ", "  a  b  ", "a\tb", "a \t b", "\ta\t",
              "a\nb\n", "\n a", "a\u00a0b", "a\u3000b", "a\u200bb", "\u2003", "line1\nline2\r\nline3\rline4",
              "    indented\n        more", "trailing space \n", " \n \n ", "a" + " " * 100 + "b"]:
        add("whitespace", s)

    for s in ["混合精度训练与推理", "東京で会いましょう", "日本語のテキストです。", "한국어 텍스트 입니다", "漢字　全角　スペース",
              "你好，世界！", "测试 test 测试", "ｆｕｌｌｗｉｄｔｈ ＡＢＣ １２３", "𠮷野家", "𪚥"]:
        add("cjk", s)
    for s in ["नमस्ते दुनिया", "हिन्दी भाषा में पाठ", "क्षत्रिय ज्ञान", "क्‍ष", "संस्कृतम्", "१२३४५"]:
        add("devanagari", s)
    for s in ["مرحبا بالعالم", "اللغة العربية", "النص مع English و١٢٣", "\u202bمرحبا\u202c", "ـــ", "بِسْمِ اللَّهِ"]:
        add("arabic", s)
    for s in ["สวัสดีชาวโลก", "ภาษาไทยไม่มีช่องว่าง", "ทดสอบ test ทดสอบ", "\u0e01\u0e34\u0e49", "ๆ"]:
        add("thai", s)
    for s in ["Привет мир", "Γειά σου Κόσμε", "שלום עולם", "ñ ü é ß ø å", "ǆ ǈ ǌ ǅ", "İstanbul ıi", "ΐ"]:
        add("other_scripts", s)
    for s in ["e\u0301", "e\u0301\u0323", "a\u0308\u0308", "\u0301", "\u0301a", "n\u0303", "Å", "A\u030a", "ﬁ ﬂ",
              "\u1100\u1161\u11a8", "\u0627\u064b", "a\u200db", "a\ufe0f", "\u20e3", "x\u0300\u0301\u0302\u0303\u0304"]:
        add("combining", s)
    for s in ["😀", "😀😀", "🚀🔥", "👍🏽", "👨\u200d👩\u200d👧\u200d👦", "🏳️\u200d🌈", "🇯🇵", "🇺🇸🇯🇵🇩🇪", "1️⃣", "#️⃣", "❤️",
              "🧑🏿\u200d💻", "\U0001faf6", "text 😀 more 🚀 text", "🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
              "☺", "☺️", "\u2764", "👩\u200d❤️\u200d💋\u200d👨"]:
        add("emoji", s)
    for c in CONTROL:
        add("control", "a" + c + "b")
    add("control", "".join(CONTROL))
    add("control", "\x00")
    add("control", "\x00\x00\x00")
    add("control", "\x1b[31mred\x1b[0m")
    for s in PRIVATE + ["\ue000\ue001\ue002", "a\ue000b", "\U000f0000\U000f0001", "\U0010ffff\U0010ffff"]:
        add("private_use", s)
    for ch in absent_chars:
        add("byte_fallback", ch)
        add("byte_fallback", "a" + ch + "b")
    add("byte_fallback", "".join(absent_chars[:40]))
    add("byte_fallback", " ".join(absent_chars[:40]))
    add("byte_fallback", "\ufffd")
    add("byte_fallback", "\ufffe\uffff")

    pieces = ["<bos>", "<eos>", "<pad>", "<unk>", "<mask>", "<|turn>", "<turn|>", "<|tool>", "<tool|>", "<|tool_call>",
              "<tool_call|>", "<|tool_response>", "<tool_response|>", '<|"|>', "<|think|>", "<|channel>", "<channel|>",
              "<|image>", "<image|>", "<|audio>", "<audio|>"]
    for p in pieces:
        add("special_strings", p)
        add("special_strings", "hello " + p + " world")
        add("special_strings", p + p)
        add("special_strings", "x" + p)
        add("special_strings", p + "x")
        add("special_strings", p + " leading")
        add("special_strings", "trailing " + p)
    for s in ["<bo", "<bos", "bos>", "<|turn", "turn|>", "<", ">", "<>", "<|", "|>", "<<bos>>", "<bos>>", "<<bos>",
              "<BOS>", "<Bos>", "< bos>", "<bos >", "<b os>", "<|turn>>", "<<|turn>", "<|turn><turn|>", "<|turn>x<turn|>",
              "<unused0>", "<unused6226>", "<0x41>", "<0xE6><0x97><0xA5>", "<bos><bos><bos>", "<eos><bos>",
              "<bos>hi<eos>", "<bos> hi", "<bos>\nhi", " <bos>hi", "<bos>", "<bos><eos>", "a<bos>b<eos>c",
              "<|image|", "<|image>|", "|<|image>", "<image|>>", "<|audio", "<|channel>thought<channel|>",
              "<|\"|>x<|\"|>", "<|tool_call>call:f{}<tool_call|>"]:
        add("special_strings", s)
    for s in ["<|image|>", "a <|image|> b", "<|video|>", "x<|video|>", "<|audio|>", "<|image|><|audio|>", "<|image|>" * 3]:
        add("placeholders", s)
    add("placeholders", "<bos><|image|>")
    add("placeholders", "see <|audio|>", "query")

    for s in ["0", "00", "007", "123", "1234567890", "12345678901234567890", "0" * 40, "9" * 65, "3.14159", "-1e-3", "1,234,567",
              "2026-10-16T08:00:00Z", "+1 (555) 010-9999", "0x1F", "1/2", "١٢٣", "①②③", "x2", "a1b2c3", "v1.2.3"]:
        add("digits", s)
    for k, s in CODE.items():
        add("code", s)
        add("code", s + s)
    add("code", "    " * 20 + "x")
    add("code", "\n".join("    " * i + "line%d" % i for i in range(30)))

    add("long", long_text(100_000))
    add("long", "word " * 20_000)
    add("long", "x" * 50_000)
    add("long", "日本語" * 10_000)

    rng = random.Random(SEED)
    for _ in range(4000):
        add("fuzz", fuzz_string(rng))
    return cases


def decode_cases(tok, rendered_ids, byte_ids, vocab_size):
    """[(group, ids, skip_special)]."""
    out = []
    for ids in rendered_ids[:1500:3]:
        out.append(("roundtrip", ids, True))
        out.append(("roundtrip", ids, False))
    rng = random.Random(SEED + 2)
    specials = [0, 1, 2, 3, 4, 46, 47, 48, 105, 106, 255999, 256000, 258880, 258881, 258884]
    for _ in range(2500):
        n = rng.randrange(0, 40)
        ids = []
        for _ in range(n):
            k = rng.randrange(0, 10)
            if k < 3:
                ids.append(rng.randrange(0, vocab_size))
            elif k < 6:
                ids.append(byte_ids[rng.choice([rng.randrange(0, 256), rng.randrange(0xC2, 0xF5), rng.randrange(0x80, 0xC0)])])
            elif k < 8:
                ids.append(rng.choice(specials))
            elif k < 9:
                ids.append(rng.choice([vocab_size, vocab_size + 1, 1 << 20, 1 << 31, 4294967294]))
            else:
                ids.append(tok.token_to_id("\u2581") if rng.randrange(0, 2) else tok.token_to_id("\u2581a"))
        out.append(("random", ids, bool(rng.randrange(0, 2))))
    seqs = {
        "valid2": [0xC3, 0xA9], "valid3": [0xE6, 0x97, 0xA5], "valid4": [0xF0, 0x9F, 0x98, 0x80],
        "short3": [0xE6, 0x97], "short4": [0xF0, 0x9F, 0x98], "stray_cont": [0x80], "stray_cont2": [0x41, 0x80, 0x42],
        "overlong": [0xC0, 0x80], "surrogate": [0xED, 0xA0, 0x80], "too_big": [0xF4, 0x90, 0x80, 0x80],
        "ff": [0xFF], "valid_then_bad": [0xC3, 0xA9, 0xC3], "two_valid": [0xC3, 0xA9, 0xE6, 0x97, 0xA5],
        "nul": [0x00], "all": list(range(256)),
    }
    for name, bs in seqs.items():
        out.append(("bytes", [byte_ids[b] for b in bs], True))
        out.append(("bytes", [byte_ids[b] for b in bs[:1]] + [tok.token_to_id("a")] + [byte_ids[b] for b in bs[1:]], True))
    out.append(("bytes", [], True))
    out.append(("bytes", [1], True))
    out.append(("bytes", [1], False))
    return out


# ---------------------------------------------------------------- writing

def write_corpus(out, cases, prompt_text, oracle):
    """oracle(prompt_text_or_None, text) -> (status, st_ids, raw_ids, plain_ids)."""
    os.makedirs(out, exist_ok=True)
    texts = [c[2] for c in cases]
    offsets = write_blob(out, "texts", texts)
    save_i64(os.path.join(out, "text_offsets.npy"), offsets)
    prompts = sorted({c[1] for c in cases if c[1] is not None})
    write_lines(out, "prompts.txt", prompts if prompts else ["-"])
    save_i64(os.path.join(out, "prompt_idx.npy"), [prompts.index(c[1]) if c[1] is not None else -1 for c in cases])
    groups, gidx = index_of([c[0] for c in cases])
    write_lines(out, "groups.txt", groups)
    save_i64(os.path.join(out, "group_idx.npy"), gidx)
    status, st, raw, plain = [], [], [], []
    for group, pname, text in cases:
        s, a, b, c = oracle(prompt_text.get(pname) if pname else None, text)
        status.append(s)
        st.append(a)
        raw.append(b)
        plain.append(c)
    save_i64(os.path.join(out, "status.npy"), status)
    write_ragged(out, "st_ids", st)
    write_ragged(out, "raw_ids", raw)
    write_ragged(out, "plain_ids", plain)
    return status


def write_decodes(out, tok, decs):
    outs = [tok.decode(ids, skip_special_tokens=skip) for _, ids, skip in decs]
    offs = write_blob(out, "dec_out", outs)
    save_i64(os.path.join(out, "dec_out_offsets.npy"), offs)
    write_ragged(out, "dec", [ids for _, ids, _ in decs])
    os.replace(os.path.join(out, "dec.npy"), os.path.join(out, "dec_ids.npy"))
    os.replace(os.path.join(out, "dec_offsets.npy"), os.path.join(out, "dec_id_offsets.npy"))
    save_i64(os.path.join(out, "dec_skip.npy"), [int(skip) for _, _, skip in decs])
    groups, gidx = index_of([d[0] for d in decs])
    write_lines(out, "dec_groups.txt", groups)
    save_i64(os.path.join(out, "dec_group_idx.npy"), gidx)


def byte_token_ids(tok):
    ids = [tok.token_to_id("<0x%02X>" % b) for b in range(256)]
    assert all(i is not None for i in ids)
    return ids


# ---------------------------------------------------------------- tiny

TINY_PROMPTS = {}
TINY_TEXTS = [
    "", " ", "hello", "hello world", " hello  world ", "hello world aab", "aab aab", "llll", "lllll", "ll ll",
    "the world", "thethe", "a", "ab", "abc", "ba", "aaa", "aaaa", "abab", "ab ab  ab", "  a", "a  ", "x", "xyz",
    "<bos>hello<eos>", "<a>", "<a><b>", "<a><b><a>", "<a<b>", "<a><", "<b><a>", "<|turn>hello<|turn>", "<|tur",
    "日😀", "日😀日", "😀日", "日日😀😀日", "x<unk>y", "<pad><pad>", "[X]", "a[X]b", "a [X] b", "<end>", "<end>a", "a<end>", "日本", "日é日", "é日é", "😀 hello 😀",
    "\t", "a\tb", "\t\t", "\ud7ff\ue000", "\u2581", "\u2581hello", "hello\u2581world", "\n", "a\nb", "\x00", "\x7f\x80",
    "e\u0301", "\u0301", "日é日 ab\t",
]


def tiny_corpus():
    rng = random.Random(SEED + 3)
    alphabet = list("abcdehilorstw. \t\n") + ["é", "日", "😀", "<", ">", "<a>", "<b>", "[X]", "<end>", "<bos>", "\u2581"]
    texts = list(TINY_TEXTS)
    for _ in range(600):
        texts.append("".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 25))))
    return texts


def run_tiny():
    for name in ("tiny_gemma", "tiny_legacy", "tiny_bytes"):
        path = os.path.join(TINY_DIR, name + ".json")
        tok = Tokenizer.from_file(path)
        out = os.path.join(TINY_DIR, name)
        cases = [("tiny", None, t) for t in tiny_corpus()]

        def oracle(prompt, text):
            raw = tok.encode(text, add_special_tokens=True).ids
            plain = tok.encode(text, add_special_tokens=False).ids
            return 0, raw, raw, plain

        write_corpus(out, cases, {}, oracle)
        n = tok.get_vocab_size()
        rng = random.Random(SEED + 4)
        decs = []
        for text in tiny_corpus()[:200]:
            ids = tok.encode(text).ids
            decs.append(("roundtrip", ids, True))
            decs.append(("roundtrip", ids, False))
        for _ in range(400):
            ids = [rng.choice([rng.randrange(0, n), rng.randrange(0, n), n, n + 5, 1 << 20]) for _ in range(rng.randrange(0, 12))]
            decs.append(("random", ids, bool(rng.randrange(0, 2))))
        write_decodes(out, tok, decs)
        print(f"{name}: {len(cases)} encode cases, {len(decs)} decode cases -> {out}")


# ---------------------------------------------------------------- real

def snapshot():
    if len(sys.argv) > 2:
        return sys.argv[2]
    hits = sorted(__import__("glob").glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/*/config.json")))
    if not hits:
        sys.exit("no google/embeddinggemma-2 snapshot; pass its directory")
    return os.path.dirname(hits[-1])


def run_real():
    import torch
    from sentence_transformers import SentenceTransformer

    snap = snapshot()
    out = os.environ.get("EMBEDGEMMA2_TOK_REF_DIR", os.path.join(ROOT, "target", "embedgemma2_tok_ref"))
    tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
    model = SentenceTransformer(snap, device="cpu", model_kwargs={"dtype": torch.float32})
    tf = model[0]
    prompts = dict(model.prompts)
    assert BOS == json.load(open(os.path.join(snap, "tokenizer_config.json")))["bos_token"]
    n_vocab = tok.get_vocab_size()
    byte_ids = byte_token_ids(tok)

    # Characters with no vocabulary entry of their own (they take the byte fallback).
    vocab = tok.get_vocab()
    absent = [c for c in RARE + [chr(c) for c in range(0xe000, 0xe040)] + [chr(c) for c in (0x10fffd, 0xfdd0, 0x1fffe)]
              if c not in vocab]
    absent = absent[:60]

    def oracle(prompt, text):
        rendered = (prompt or "") + text
        skip_specials = rendered.startswith(BOS)
        lib = tok.encode(rendered, add_special_tokens=not skip_specials).ids
        raw = tok.encode(rendered, add_special_tokens=True).ids
        plain = tok.encode(rendered, add_special_tokens=False).ids
        try:
            feats = tf.preprocess([text], prompt=prompt)
        except ValueError as e:
            assert any(p in text for p in ("<|image|>", "<|video|>", "<|audio|>")), (text, e)
            return 1, lib, raw, plain
        mask = feats["attention_mask"][0]
        st = feats["input_ids"][0][: int(mask.sum())].tolist()
        assert st == lib, f"oracle disagrees with itself on {text!r}: {st[:10]} vs {lib[:10]}"
        return 0, st, raw, plain

    cases = build_cases(prompts, absent)
    status = write_corpus(out, cases, prompts, oracle)
    # Decode inputs: the corpus's own ids, then random and byte sequences.
    st_ids = [tok.encode((prompts.get(c[1], "") if c[1] else "") + c[2]).ids for c in cases]
    decs = decode_cases(tok, st_ids, byte_ids, n_vocab)
    write_decodes(out, tok, decs)

    # The batch quirk the Rust tokenizer deliberately does not reproduce.
    def batch(texts, prompt=None):
        f = tf.preprocess(texts, prompt=prompt)
        m = f["attention_mask"]
        return [f["input_ids"][i][: int(m[i].sum())].tolist() for i in range(len(texts))]

    quirk = {
        "first_bos": batch(["<bos>hi", "plain"]),
        "second_bos": batch(["plain", "<bos>hi"]),
        "alone": [batch([t])[0] for t in ("<bos>hi", "plain")],
    }
    assert quirk["first_bos"][1] == tok.encode("plain", add_special_tokens=False).ids, "transformers fixed the batch quirk"
    assert quirk["second_bos"][0] == tok.encode("plain").ids
    with open(os.path.join(out, "batch_quirk.json"), "w") as f:
        json.dump(quirk, f)

    prov = {
        "generator": "scripts/gen_embedgemma2_tok_fixtures.py",
        "snapshot": snap,
        "seed": SEED,
        "cases": len(cases),
        "decode_cases": len(decs),
        "refused_cases": int(sum(status)),
        "absent_chars_used": len(absent),
        "versions": {p: importlib.metadata.version(p)
                     for p in ("torch", "numpy", "tokenizers", "transformers", "sentence-transformers")},
    }
    with open(os.path.join(out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "tiny":
        run_tiny()
    elif mode == "real":
        run_real()
    else:
        sys.exit(__doc__)
