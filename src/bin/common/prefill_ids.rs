//! The ids file every prefill+score lane reads (`tests/fixtures/qwen35_prefill_ids.json`).
//!
//! One file, three readers: `bench_qwen35_layers --prefill-score` (this module),
//! `bench/qwen35_prefill_torch.py` and `bench/qwen35_prefill_mlx.py` (through
//! `bench/prefill_ids.py`, which owns the writer and the same contract). A lane
//! at length `T` runs `ids[..T]`, so every side sees the same tokens at every
//! length, and the file lists the FNV-1a 32 digest of each prefix it was built
//! for (`digests`). A length the file has no digest for, a digest that does not
//! match, a token at or above the vocabulary or a file shorter than `T` is an
//! error, never a shorter run.
//!
//! ```text
//! {"schema": "tessl-prefill-ids-v1", "snapshot": "...", "vocab": 248320,
//!  "ids": [u32; N], "answers": [u32; 17], "digests": {"200": u32, ...}}
//! ```

use crate::json::{self, Json, NumForm, Syntax};

const SYNTAX: Syntax = Syntax {
    what: "prefill ids file",
    max_depth: 3,
    uints_only: true,
    literals: false,
};

pub const SCHEMA: &str = "tessl-prefill-ids-v1";

/// A parsed ids file.
#[derive(Debug, PartialEq)]
pub struct PrefillIds {
    pub vocab: u32,
    pub ids: Vec<u32>,
    pub answers: Vec<u32>,
    pub digests: Vec<(usize, u32)>,
}

/// FNV-1a 32 over the ids as little-endian u32 words; `bench/prefill_ids.py`
/// computes the same number.
pub fn fnv1a32(ids: &[u32]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for id in ids {
        for b in id.to_le_bytes() {
            h = (h ^ u32::from(b)).wrapping_mul(0x0100_0193);
        }
    }
    h
}

fn uint_array(doc: &Json, key: &str) -> Result<Vec<u32>, String> {
    let Some(Json::Array(a)) = doc.get(key) else {
        return Err(format!("prefill ids file: \"{key}\" is not an array"));
    };
    a.iter()
        .map(|v| match v {
            Json::Num {
                int,
                form: NumForm::Uint,
                ..
            } => u32::try_from(*int).map_err(|_| format!("prefill ids file: \"{key}\" holds {int}, above u32")),
            _ => Err(format!("prefill ids file: \"{key}\" holds a non-integer")),
        })
        .collect()
}

impl PrefillIds {
    pub fn parse(text: &str) -> Result<Self, String> {
        let doc = json::parse(text, SYNTAX)?;
        match doc.get("schema") {
            Some(Json::Str(s)) if s == SCHEMA => {}
            other => return Err(format!("prefill ids file: schema {other:?}, expected {SCHEMA:?}")),
        }
        let vocab = match doc.get("vocab") {
            Some(v) => v
                .exact_int()
                .and_then(|n| u32::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or("prefill ids file: \"vocab\" is not a positive u32")?,
            None => return Err("prefill ids file: no \"vocab\"".into()),
        };
        let (ids, answers) = (uint_array(&doc, "ids")?, uint_array(&doc, "answers")?);
        if answers.is_empty() {
            return Err("prefill ids file: no answers".into());
        }
        if let Some(&bad) = ids.iter().chain(&answers).find(|&&i| i >= vocab) {
            return Err(format!("prefill ids file: token {bad} >= vocab {vocab}"));
        }
        let Some(Json::Object(kv)) = doc.get("digests") else {
            return Err("prefill ids file: \"digests\" is not an object".into());
        };
        let mut digests = Vec::new();
        for (k, v) in kv {
            let len: usize = k.parse().map_err(|_| format!("prefill ids file: digest key {k:?} is not a length"))?;
            let d = v
                .exact_int()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| format!("prefill ids file: digest of {k} is not a u32"))?;
            digests.push((len, d));
        }
        Ok(Self {
            vocab,
            ids,
            answers,
            digests,
        })
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("--ids {}: {e}", path.display()))?;
        Self::parse(&text).map_err(|e| format!("--ids {}: {e}", path.display()))
    }

    /// The first `t` ids, with the file's digest of that prefix checked. Fails
    /// when the file is shorter than `t`, lists no digest for `t`, or the
    /// digest disagrees.
    pub fn prefix(&self, t: usize) -> Result<&[u32], String> {
        if t == 0 || t > self.ids.len() {
            return Err(format!("length {t} asked of an ids file with {} ids", self.ids.len()));
        }
        let want = self
            .digests
            .iter()
            .find(|(len, _)| *len == t)
            .map(|(_, d)| *d)
            .ok_or_else(|| format!("the ids file lists no digest for length {t}; regenerate it with that length"))?;
        let got = fnv1a32(&self.ids[..t]);
        if got != want {
            return Err(format!("length {t}: ids digest {got} but the file says {want}"));
        }
        Ok(&self.ids[..t])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(ids: &[u32], digests: &str) -> String {
        let list = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        format!(
            "{{\"schema\":\"{SCHEMA}\",\"vocab\":100,\"ids\":[{list}],\"answers\":[3,4],\"digests\":{{{digests}}}}}"
        )
    }

    #[test]
    fn fnv_agrees_with_the_python_writer() {
        // The same numbers `bench/test_comparator_lanes.py` pins from `prefill_lane.fnv1a32`.
        assert_eq!(fnv1a32(&[]), 2_166_136_261);
        assert_eq!(fnv1a32(&[1, 2, 3]), 2_034_659_765);
        assert_eq!(fnv1a32(&[248_319]), 3_941_604_598);
    }

    #[test]
    fn the_committed_ids_file_parses_and_every_listed_length_verifies() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_prefill_ids.json");
        let f = PrefillIds::load(&path).unwrap();
        assert_eq!((f.vocab, f.answers.len(), f.ids.len()), (248_320, 17, 8192));
        for t in [200, 1024, 2048, 8192] {
            assert_eq!(f.prefix(t).unwrap().len(), t);
        }
    }

    #[test]
    fn a_prefix_needs_its_digest_and_enough_ids() {
        let ids = [5u32, 6, 7, 8];
        let d2 = fnv1a32(&ids[..2]);
        let f = PrefillIds::parse(&doc(&ids, &format!("\"2\":{d2},\"4\":1"))).unwrap();
        assert_eq!(f.prefix(2).unwrap(), &ids[..2]);
        assert!(f.prefix(4).unwrap_err().contains("digest"));
        assert!(f.prefix(3).unwrap_err().contains("no digest"));
        assert!(f.prefix(5).unwrap_err().contains("4 ids"));
        assert!(f.prefix(0).is_err());
    }

    #[test]
    fn a_token_outside_the_vocabulary_is_refused() {
        assert!(PrefillIds::parse(&doc(&[1, 100], "")).unwrap_err().contains(">= vocab"));
    }

    #[test]
    fn a_wrong_schema_or_a_negative_id_is_refused() {
        assert!(PrefillIds::parse("{\"schema\":\"x\"}").is_err());
        let neg = doc(&[1], "").replace("[1]", "[-1]");
        assert!(PrefillIds::parse(&neg).is_err());
    }
}
