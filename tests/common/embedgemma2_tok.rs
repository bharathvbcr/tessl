//! Reader for the (text, ids) corpora `scripts/gen_embedgemma2_tok_fixtures.py`
//! writes, and a per-group tally for the parity tests. The file layout is in
//! that script's docstring.

use std::collections::BTreeMap;
use std::path::Path;

use tessl::npy::read_npy;

pub struct Corpus {
    pub texts: Vec<String>,
    /// The prompt name of each text, if it has one.
    pub prompt: Vec<Option<String>>,
    pub group: Vec<String>,
    /// 0: sentence-transformers accepts the text; 1: its processor refuses it.
    pub status: Vec<i64>,
    pub st_ids: Vec<Vec<u32>>,
    pub raw_ids: Vec<Vec<u32>>,
    pub plain_ids: Vec<Vec<u32>>,
    pub dec_ids: Vec<Vec<u32>>,
    pub dec_skip: Vec<bool>,
    pub dec_out: Vec<String>,
    pub dec_group: Vec<String>,
}

fn i64s(dir: &Path, name: &str) -> Vec<i64> {
    let p = dir.join(name);
    read_npy(&p)
        .unwrap_or_else(|e| panic!("{}: {e} (run scripts/gen_embedgemma2_tok_fixtures.py)", p.display()))
        .i64_slice()
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .to_vec()
}

fn lines(dir: &Path, name: &str) -> Vec<String> {
    let p = dir.join(name);
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .lines()
        .map(str::to_string)
        .collect()
}

fn blob(dir: &Path, bin: &str, offsets: &str) -> Vec<String> {
    let p = dir.join(bin);
    let data = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let off = i64s(dir, offsets);
    off.windows(2)
        .map(|w| String::from_utf8(data[w[0] as usize..w[1] as usize].to_vec()).expect("utf-8 corpus text"))
        .collect()
}

/// A ragged int array: `<name>.npy` flat, `<offsets>` the N+1 row bounds.
fn ragged(dir: &Path, name: &str, offsets: &str) -> Vec<Vec<u32>> {
    let flat = i64s(dir, name);
    let off = i64s(dir, offsets);
    off.windows(2)
        .map(|w| {
            flat[w[0] as usize..w[1] as usize]
                .iter()
                .map(|&v| u32::try_from(v).expect("id fits u32"))
                .collect()
        })
        .collect()
}

impl Corpus {
    pub fn load(dir: &Path) -> Corpus {
        let texts = blob(dir, "texts.bin", "text_offsets.npy");
        let prompts = lines(dir, "prompts.txt");
        let groups = lines(dir, "groups.txt");
        let dec_groups = lines(dir, "dec_groups.txt");
        let c = Corpus {
            prompt: i64s(dir, "prompt_idx.npy")
                .into_iter()
                .map(|i| usize::try_from(i).ok().map(|i| prompts[i].clone()))
                .collect(),
            group: i64s(dir, "group_idx.npy")
                .into_iter()
                .map(|i| groups[i as usize].clone())
                .collect(),
            status: i64s(dir, "status.npy"),
            st_ids: ragged(dir, "st_ids.npy", "st_ids_offsets.npy"),
            raw_ids: ragged(dir, "raw_ids.npy", "raw_ids_offsets.npy"),
            plain_ids: ragged(dir, "plain_ids.npy", "plain_ids_offsets.npy"),
            dec_ids: ragged(dir, "dec_ids.npy", "dec_id_offsets.npy"),
            dec_skip: i64s(dir, "dec_skip.npy").into_iter().map(|v| v != 0).collect(),
            dec_out: blob(dir, "dec_out.bin", "dec_out_offsets.npy"),
            dec_group: i64s(dir, "dec_group_idx.npy")
                .into_iter()
                .map(|i| dec_groups[i as usize].clone())
                .collect(),
            texts,
        };
        let n = c.texts.len();
        for (what, len) in [
            ("prompt", c.prompt.len()),
            ("group", c.group.len()),
            ("status", c.status.len()),
            ("st_ids", c.st_ids.len()),
            ("raw_ids", c.raw_ids.len()),
            ("plain_ids", c.plain_ids.len()),
        ] {
            assert_eq!(len, n, "{what} rows vs {n} texts");
        }
        let m = c.dec_ids.len();
        for (what, len) in [
            ("dec_skip", c.dec_skip.len()),
            ("dec_out", c.dec_out.len()),
            ("dec_group", c.dec_group.len()),
        ] {
            assert_eq!(len, m, "{what} rows vs {m} decode cases");
        }
        c
    }
}

/// Per-group pass counts, printed as the parity report.
#[derive(Default)]
pub struct Tally {
    groups: BTreeMap<String, (usize, usize)>,
    failures: Vec<String>,
}

impl Tally {
    pub fn record(&mut self, group: &str, ok: bool, describe: impl FnOnce() -> String) {
        let e = self.groups.entry(group.to_string()).or_default();
        e.0 += 1;
        if ok {
            e.1 += 1;
        } else if self.failures.len() < 12 {
            self.failures.push(format!("[{group}] {}", describe()));
        }
    }

    /// Print one line per group and fail if any case differed.
    pub fn finish(self, what: &str) {
        let (mut total, mut passed) = (0, 0);
        for (g, (n, ok)) in &self.groups {
            eprintln!("{what} {g:<16} {ok}/{n}");
            total += n;
            passed += ok;
        }
        eprintln!("{what} total {passed}/{total}");
        assert!(
            self.failures.is_empty() && passed == total,
            "{what}: {} of {total} cases differ; the first:\n{}",
            total - passed,
            self.failures.join("\n")
        );
    }
}

/// The `(prompt_name, text)` pairs of `tools/embedgemma2_ref/make_reference.py`'s
/// `texts.json` (`{"texts": [{"prompt_name": ..., "text": ...}, ...]}`, written
/// with `ensure_ascii=False`). Only that shape is read: a string is scanned with
/// the RFC 8259 escapes (surrogate pairs included) and anything else is a panic.
pub fn reference_texts(json: &str) -> Vec<(String, String)> {
    fn string_after(s: &str, key: &str, from: usize) -> Option<(String, usize)> {
        let k = s[from..].find(key)? + from + key.len();
        let rest = &s[k..];
        let open = rest.find('"')? + k + 1;
        let mut out = String::new();
        let mut chars = s[open..].char_indices();
        while let Some((i, c)) = chars.next() {
            match c {
                '"' => return Some((out, open + i + 1)),
                '\\' => match chars.next().expect("an escape").1 {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'b' => out.push('\u{8}'),
                    'f' => out.push('\u{c}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => {
                        let hex = |it: &mut std::str::CharIndices<'_>| -> u32 {
                            let h: String = it.take(4).map(|(_, c)| c).collect();
                            u32::from_str_radix(&h, 16).expect("\\u escape")
                        };
                        let hi = hex(&mut chars);
                        let cp = if (0xD800..0xDC00).contains(&hi) {
                            assert_eq!(chars.next().map(|x| x.1), Some('\\'));
                            assert_eq!(chars.next().map(|x| x.1), Some('u'));
                            0x10000 + ((hi - 0xD800) << 10) + (hex(&mut chars) - 0xDC00)
                        } else {
                            hi
                        };
                        out.push(char::from_u32(cp).expect("a scalar value"));
                    }
                    other => panic!("escape \\{other}"),
                },
                c => out.push(c),
            }
        }
        None
    }
    let mut at = json.find("\"texts\"").expect("a \"texts\" array");
    let mut out = Vec::new();
    while let Some((prompt, next)) = string_after(json, "\"prompt_name\":", at) {
        let (text, next) = string_after(json, "\"text\":", next).expect("a text after its prompt_name");
        out.push((prompt, text));
        at = next;
    }
    assert!(!out.is_empty(), "no texts found");
    out
}

/// `{:?}` of at most the first 80 characters, so a failure over a 100k-char text stays readable.
pub fn brief(s: &str) -> String {
    let head: String = s.chars().take(80).collect();
    if head.len() < s.len() {
        format!("{head:?}... ({} bytes)", s.len())
    } else {
        format!("{head:?}")
    }
}

/// The first position where two id lists differ, with a little context.
pub fn first_diff(got: &[u32], want: &[u32]) -> String {
    match got.iter().zip(want).position(|(a, b)| a != b) {
        Some(i) => format!(
            "differ at {i} (got {} ids, want {}): got {:?} want {:?}",
            got.len(),
            want.len(),
            &got[i.saturating_sub(2)..(i + 3).min(got.len())],
            &want[i.saturating_sub(2)..(i + 3).min(want.len())]
        ),
        None => format!(
            "got {} ids, want {} (one is a prefix of the other)",
            got.len(),
            want.len()
        ),
    }
}
