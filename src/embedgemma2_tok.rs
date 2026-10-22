//! EmbeddingGemma 2's tokenizer, in Rust: what `tokenizers` (the Hugging Face
//! library) and sentence-transformers turn a text into, so no caller needs a
//! Python tokenizer to feed [`crate::embedgemma2`].
//!
//! The snapshot's `tokenizer.json` is a SentencePiece-style BPE with byte
//! fallback and no pre-tokenization in effect: the normalizer rewrites every
//! space to `U+2581`, the one pre-tokenizer (a split on a space) then finds
//! nothing, and the BPE merges run over each stretch of text between added
//! tokens as one word. The added tokens (`<bos>`, `<|turn>`, ...) are matched
//! in the raw text before any of that. `docs/embedgemma2.md` (section
//! "Tokenizer") records what each piece does and how it was checked.
//!
//! # What is implemented
//!
//! The parts of the `tokenizer.json` schema this model family uses, each
//! checked against the `tokenizers` Python library: a BPE model (`vocab`,
//! `merges` as pairs or as `"a b"` strings, `byte_fallback`, `unk_token`,
//! `fuse_unk`, `ignore_merges`), a `Replace` normalizer (or a `Sequence` of
//! them), a literal `Split` pre-tokenizer in the `MergedWithPrevious` mode,
//! added tokens matched in the raw text (`normalized: false`, no strip, no
//! `single_word`), a `TemplateProcessing` post-processor for a single
//! sequence, and a decoder of `Replace`, `ByteFallback` and `Fuse` steps.
//! Anything else (a dropout, a continuing-subword prefix, a regex pattern,
//! `lstrip`, a truncation or padding block, a `ByteLevel` or `Metaspace`
//! step, an added token outside the vocabulary) is refused by name when the
//! file is loaded, never approximated. The post-processor's pair template is
//! not read: the model is only ever given one text.
//!
//! # The ids the model receives
//!
//! [`Tokenizer::encode`] and its siblings reproduce sentence-transformers
//! (6.1) for `google/embeddinggemma-2`, which is not the same as
//! `Tokenizer.encode(text)` on the library alone:
//!
//! * a prompt (`encode(prompt_name=...)`) is a system message that the chat
//!   template writes in front of the text, so the tokenizer sees
//!   `prompt + text`;
//! * `<bos>` and `<eos>` are added around that, **unless** it starts with the
//!   `bos_token` of `tokenizer_config.json`: then none are added, so the text's
//!   own `<bos>` is the only one and there is no `<eos>`
//!   (`ProcessorMixin.apply_chat_template`, `processing_utils.py`);
//! * a text that holds `<|image|>`, `<|video|>` or `<|audio|>` is refused,
//!   because the processor refuses a placeholder with no media to fill it.
//!
//! In a batch, transformers applies the `<bos>` rule to the **first** text's
//! rendering for the whole batch. That makes a text's ids depend on its
//! neighbours, so this module applies the rule to each text on its own (as a
//! batch of one): [`Tokenizer::encode_batch`] returns what [`Tokenizer::encode`]
//! returns for each text, and differs from sentence-transformers only for a
//! batch whose first text starts with `<bos>`.
//!
//! # Load
//!
//! `tokenizer.json` (32 MB for this vocabulary) is parsed once, with the
//! crate's strict JSON parser, into compact tables: the token text in one
//! arena, an open-addressing index from text to id, a character table, and an
//! open-addressing table of the merge ranks. Encoding allocates only its
//! working buffers and never reads the JSON again.

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::Read;
use std::path::Path;

use crate::json::{self, Json, NumForm, Syntax};

/// Largest `tokenizer.json` read, in bytes (the checkpoint's is 32 MB).
pub const MAX_TOKENIZER_BYTES: u64 = 256 << 20;
/// Largest sidecar config (`tokenizer_config.json`,
/// `config_sentence_transformers.json`) read, in bytes.
pub const MAX_CONFIG_BYTES: u64 = 1 << 20;
/// Texts of this many bytes or more are refused ([`Tokenizer::encode_with_prompt`])
/// or panic ([`Tokenizer::encode_raw`]): the model's context is 8192 tokens.
pub const MAX_TEXT_BYTES: usize = 1 << 28;

const NONE: u32 = u32::MAX;

const TOKENIZER_SYNTAX: Syntax = Syntax {
    what: "tokenizer.json",
    max_depth: 6,
    uints_only: true,
    literals: true,
};

const CONFIG_SYNTAX: Syntax = Syntax {
    what: "tokenizer config",
    max_depth: 4,
    uints_only: false,
    literals: true,
};

/// One added token: matched in the raw text, never split by the BPE.
struct Added {
    bytes: Vec<u8>,
    id: u32,
}

/// What the decoder does to the list of token strings, in order.
enum DecodeStep {
    Replace { from: String, to: String },
    ByteFallback,
    Fuse,
}

/// One item of the post-processor's template for a single sequence.
enum Template {
    Special(Vec<u32>),
    Sequence,
}

/// An open-addressing table from a token's text to its id.
struct VocabIndex {
    slots: Vec<u32>,
    mask: usize,
}

fn hash_bytes(b: &[u8]) -> u64 {
    const K: u64 = 0x517c_c1b7_2722_0a95;
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ b.len() as u64;
    let mut chunks = b.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        h = (h.rotate_left(5) ^ w).wrapping_mul(K);
    }
    for &x in chunks.remainder() {
        h = (h.rotate_left(5) ^ u64::from(x)).wrapping_mul(K);
    }
    h ^ (h >> 29)
}

/// One slot of the merge table: the pair, its rank, and the merged token.
#[derive(Clone, Copy)]
struct MergeSlot {
    key: u64,
    rank: u32,
    new: u32,
}

/// The merge table: `(left id, right id) -> (rank, merged id)`, in open
/// addressing with the key and value in one slot (one cache line per probe),
/// behind a bit filter small enough for the cache that answers most absent
/// pairs (nearly every adjacent pair of a text is absent) without touching the
/// table.
struct MergeTable {
    slots: Vec<MergeSlot>,
    shift: u32,
    filter: Vec<u64>,
    filter_shift: u32,
}

const EMPTY_KEY: u64 = u64::MAX;

impl MergeTable {
    fn with_capacity(entries: usize) -> Self {
        let slots = (entries.max(1) * 2).next_power_of_two().max(16);
        // 16 bits per entry, at least 64.
        let bits = (entries.max(1) * 16).next_power_of_two().max(64);
        MergeTable {
            slots: vec![
                MergeSlot {
                    key: EMPTY_KEY,
                    rank: 0,
                    new: 0
                };
                slots
            ],
            shift: 64 - slots.trailing_zeros(),
            filter: vec![0; bits / 64],
            filter_shift: 64 - bits.trailing_zeros(),
        }
    }

    #[inline]
    fn slot(&self, key: u64) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    #[inline]
    fn filter_bit(&self, key: u64) -> usize {
        (key.wrapping_mul(0xD6E8_FEB8_6659_FD93) >> self.filter_shift) as usize
    }

    #[inline]
    fn key(a: u32, b: u32) -> u64 {
        (u64::from(a) << 32) | u64::from(b)
    }

    /// Insert; a pair already present is overwritten, as the library's
    /// `HashMap` collect does (the later merge line wins).
    fn insert(&mut self, a: u32, b: u32, rank: u32, new: u32) {
        let key = Self::key(a, b);
        let bit = self.filter_bit(key);
        self.filter[bit / 64] |= 1 << (bit % 64);
        let mask = self.slots.len() - 1;
        let mut i = self.slot(key);
        loop {
            let s = &mut self.slots[i];
            if s.key == EMPTY_KEY || s.key == key {
                *s = MergeSlot { key, rank, new };
                return;
            }
            i = (i + 1) & mask;
        }
    }

    #[inline]
    fn get(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        let key = Self::key(a, b);
        let bit = self.filter_bit(key);
        if self.filter[bit / 64] & (1 << (bit % 64)) == 0 {
            return None;
        }
        let mask = self.slots.len() - 1;
        let mut i = self.slot(key);
        loop {
            let s = self.slots[i];
            if s.key == key {
                return Some((s.rank, s.new));
            }
            if s.key == EMPTY_KEY {
                return None;
            }
            i = (i + 1) & mask;
        }
    }
}

/// A BPE symbol in the merge list (the library's `Symbol`).
#[derive(Clone, Copy)]
struct Symbol {
    c: u32,
    prev: i32,
    next: i32,
    len: u32,
}

/// The EmbeddingGemma 2 tokenizer. Immutable after loading, so one value is
/// shared by every thread.
pub struct Tokenizer {
    arena: Vec<u8>,
    /// `(start, end)` of token `id` in `arena`.
    spans: Vec<(u32, u32)>,
    index: VocabIndex,
    /// Id of a one-character token, by character (`NONE` if there is none).
    char_ids: Vec<u32>,
    byte_ids: [u32; 256],
    merges: MergeTable,
    byte_fallback: bool,
    fuse_unk: bool,
    ignore_merges: bool,
    unk: Option<u32>,
    normalizer: Vec<(String, String)>,
    split: Option<String>,
    added: Vec<Added>,
    /// For each first byte, the added tokens that start with it, longest first.
    added_by_byte: Vec<Vec<u32>>,
    specials: Vec<u32>,
    template: Vec<Template>,
    decoder: Vec<DecodeStep>,
    bos_token: Option<String>,
    placeholders: Vec<String>,
    prompts: Vec<(String, String)>,
    /// `default_prompt_name` of `config_sentence_transformers.json`: the
    /// prompt `SentenceTransformer.encode` uses when no name is given. Always
    /// one of `prompts` when set.
    default_prompt: Option<String>,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size())
            .field("added_tokens", &self.added.len())
            .field("prompts", &self.prompts.len())
            .finish_non_exhaustive()
    }
}

fn get<'a>(obj: &'a Json, key: &str) -> Option<&'a Json> {
    obj.get(key)
}

fn as_str<'a>(j: &'a Json, what: &str) -> Result<&'a str, String> {
    match j {
        Json::Str(s) => Ok(s),
        _ => Err(format!("tokenizer.json: {what} is not a string")),
    }
}

fn as_bool(j: Option<&Json>, what: &str) -> Result<bool, String> {
    match j {
        Some(Json::Bool(b)) => Ok(*b),
        _ => Err(format!("tokenizer.json: {what} is missing or not a boolean")),
    }
}

fn as_u32(j: &Json, what: &str) -> Result<u32, String> {
    match j {
        Json::Num {
            form: NumForm::Uint,
            int: v,
            ..
        } if *v < u64::from(u32::MAX) => Ok(*v as u32),
        _ => Err(format!("tokenizer.json: {what} is not an integer below 2^32 - 1")),
    }
}

fn type_of<'a>(j: &'a Json, what: &str) -> Result<&'a str, String> {
    match get(j, "type") {
        Some(Json::Str(s)) => Ok(s),
        _ => Err(format!("tokenizer.json: {what} has no \"type\"")),
    }
}

/// `{"String": s}` (a literal pattern), the only pattern form supported.
fn string_pattern(j: Option<&Json>, what: &str) -> Result<String, String> {
    match j.and_then(|p| get(p, "String")) {
        Some(Json::Str(s)) if !s.is_empty() => Ok(s.clone()),
        Some(Json::Str(_)) => Err(format!("tokenizer.json: {what} has an empty pattern")),
        _ => Err(format!(
            "tokenizer.json: {what}: only a literal {{\"String\": ...}} pattern is supported"
        )),
    }
}

fn parse_replace(j: &Json, what: &str) -> Result<(String, String), String> {
    let from = string_pattern(get(j, "pattern"), what)?;
    let to = as_str(
        get(j, "content").ok_or_else(|| format!("tokenizer.json: {what} has no content"))?,
        what,
    )?;
    Ok((from, to.to_string()))
}

impl VocabIndex {
    fn build(arena: &[u8], spans: &[(u32, u32)]) -> Result<Self, String> {
        let slots = (spans.len().max(1) * 2).next_power_of_two().max(16);
        let mask = slots - 1;
        let mut table = vec![0u32; slots];
        for (id, &(s, e)) in spans.iter().enumerate() {
            let mut i = hash_bytes(&arena[s as usize..e as usize]) as usize & mask;
            loop {
                let cur = table[i];
                if cur == 0 {
                    table[i] = id as u32 + 1;
                    break;
                }
                let (cs, ce) = spans[cur as usize - 1];
                if arena[cs as usize..ce as usize] == arena[s as usize..e as usize] {
                    return Err("tokenizer.json: two vocabulary entries with the same text".into());
                }
                i = (i + 1) & mask;
            }
        }
        Ok(VocabIndex { slots: table, mask })
    }
}

impl Tokenizer {
    fn lookup(&self, s: &[u8]) -> Option<u32> {
        let mut i = hash_bytes(s) as usize & self.index.mask;
        loop {
            let cur = self.index.slots[i];
            if cur == 0 {
                return None;
            }
            let (cs, ce) = self.spans[cur as usize - 1];
            if &self.arena[cs as usize..ce as usize] == s {
                return Some(cur - 1);
            }
            i = (i + 1) & self.index.mask;
        }
    }

    /// The text of token `id`, or `None` for an id past the vocabulary.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        let &(s, e) = self.spans.get(id as usize)?;
        // The arena holds only whole strings copied from the JSON.
        std::str::from_utf8(&self.arena[s as usize..e as usize]).ok()
    }

    /// The id of the token whose text is `token`, if the vocabulary has it.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.lookup(token.as_bytes())
    }

    /// The number of tokens (ids are `0..vocab_size()`).
    pub fn vocab_size(&self) -> usize {
        self.spans.len()
    }

    /// Parse the text of a `tokenizer.json`. No `bos_token`, placeholders or
    /// prompts yet: add them with [`Self::with_tokenizer_config`] and
    /// [`Self::with_prompts`], or use [`Self::from_snapshot`].
    pub fn from_json(text: &str) -> Result<Self, String> {
        let root = json::parse(text, TOKENIZER_SYNTAX)?;
        Self::from_root(root)
    }

    /// [`Self::from_json`] on a file (a regular file of at most
    /// [`MAX_TOKENIZER_BYTES`]).
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let text = read_bounded(path, MAX_TOKENIZER_BYTES)?;
        Self::from_json(&text)
    }

    /// Load a Hugging Face snapshot directory: `tokenizer.json`,
    /// `tokenizer_config.json` (the `bos_token` and the three placeholder
    /// tokens) and `config_sentence_transformers.json` (the named prompts).
    /// All three must be there.
    pub fn from_snapshot(dir: &Path) -> Result<Self, String> {
        Self::load_snapshot(dir, true)
    }

    /// [`Self::from_snapshot`] for a snapshot that may have no
    /// `config_sentence_transformers.json`: without it the tokenizer has no
    /// named prompts (and no default one). A file that is there is read as
    /// strictly as ever, and a file that cannot be read for any reason but
    /// not existing is an error.
    pub fn from_snapshot_optional_prompts(dir: &Path) -> Result<Self, String> {
        Self::load_snapshot(dir, false)
    }

    fn load_snapshot(dir: &Path, prompts_required: bool) -> Result<Self, String> {
        let tok = Self::from_file(&dir.join("tokenizer.json"))?;
        let cfg = read_bounded(&dir.join("tokenizer_config.json"), MAX_CONFIG_BYTES)?;
        let tok = tok.with_tokenizer_config(&cfg)?;
        let prompts = dir.join("config_sentence_transformers.json");
        if !prompts_required {
            if let Err(e) = std::fs::symlink_metadata(&prompts) {
                if e.kind() == std::io::ErrorKind::NotFound {
                    return Ok(tok);
                }
            }
        }
        tok.with_prompts(&read_bounded(&prompts, MAX_CONFIG_BYTES)?)
    }

    /// Read `bos_token` (a string, or an added-token object with `content`)
    /// and `image_token`, `video_token`, `audio_token` from the text of a
    /// `tokenizer_config.json`. The `bos_token` is what switches the
    /// sentence-transformers rule off for a text that starts with it; each
    /// placeholder token the file names is refused in a text.
    pub fn with_tokenizer_config(mut self, text: &str) -> Result<Self, String> {
        let root = json::parse(text, CONFIG_SYNTAX)?;
        if !matches!(root, Json::Object(_)) {
            return Err("tokenizer config: the root is not an object".into());
        }
        let token = |key: &str| -> Result<Option<String>, String> {
            match get(&root, key) {
                None | Some(Json::Null) => Ok(None),
                Some(Json::Str(s)) => Ok(Some(s.clone())),
                Some(o @ Json::Object(_)) => match get(o, "content") {
                    Some(Json::Str(s)) => Ok(Some(s.clone())),
                    _ => Err(format!("tokenizer config: {key} has no string content")),
                },
                Some(_) => Err(format!("tokenizer config: {key} is not a string")),
            }
        };
        self.bos_token = token("bos_token")?.filter(|s| !s.is_empty());
        self.placeholders.clear();
        for key in ["image_token", "video_token", "audio_token"] {
            if let Some(t) = token(key)?.filter(|s| !s.is_empty()) {
                self.placeholders.push(t);
            }
        }
        Ok(self)
    }

    /// Read the named prompts (`"prompts": {name: text}`) and the
    /// `default_prompt_name` (null or absent for none, otherwise it must name
    /// one of the prompts) from the text of a `config_sentence_transformers.json`.
    pub fn with_prompts(mut self, text: &str) -> Result<Self, String> {
        let root = json::parse(text, CONFIG_SYNTAX)?;
        self.prompts.clear();
        self.default_prompt = None;
        match get(&root, "prompts") {
            None | Some(Json::Null) => {}
            Some(Json::Object(fields)) => {
                for (name, v) in fields {
                    match v {
                        Json::Str(p) => self.prompts.push((name.clone(), p.clone())),
                        _ => {
                            return Err(format!(
                                "config_sentence_transformers.json: prompt {name:?} is not a string"
                            ))
                        }
                    }
                }
            }
            Some(_) => return Err("config_sentence_transformers.json: \"prompts\" is not an object".into()),
        }
        match get(&root, "default_prompt_name") {
            None | Some(Json::Null) => {}
            Some(Json::Str(name)) => {
                if self.prompt(name).is_none() {
                    return Err(format!(
                        "config_sentence_transformers.json: default_prompt_name {name:?} is not one of its prompts"
                    ));
                }
                self.default_prompt = Some(name.clone());
            }
            Some(_) => {
                return Err("config_sentence_transformers.json: \"default_prompt_name\" is not a string or null".into())
            }
        }
        Ok(self)
    }

    /// The snapshot's `default_prompt_name`, if it names one: the prompt
    /// `SentenceTransformer.encode` applies when it is given no name. (The
    /// `encode*` methods here take the name they are given, `None` meaning no
    /// prompt; applying this default is the caller's choice.)
    pub fn default_prompt_name(&self) -> Option<&str> {
        self.default_prompt.as_deref()
    }

    /// The text of the prompt named `name`.
    pub fn prompt(&self, name: &str) -> Option<&str> {
        self.prompts.iter().find(|(n, _)| n == name).map(|(_, p)| p.as_str())
    }

    /// Every prompt name, in the file's order.
    pub fn prompt_names(&self) -> impl Iterator<Item = &str> {
        self.prompts.iter().map(|(n, _)| n.as_str())
    }

    fn from_root(root: Json) -> Result<Self, String> {
        let Json::Object(mut fields) = root else {
            return Err("tokenizer.json: the root is not an object".into());
        };
        let mut take = |key: &str| -> Option<Json> {
            fields
                .iter()
                .position(|(k, _)| k == key)
                .map(|i| fields.swap_remove(i).1)
        };
        for key in ["truncation", "padding"] {
            match take(key) {
                None | Some(Json::Null) => {}
                Some(_) => {
                    return Err(format!(
                        "tokenizer.json: a {key} block is not supported (the model feeds whole texts)"
                    ))
                }
            }
        }
        let model = take("model").ok_or("tokenizer.json: no model")?;
        let added_json = take("added_tokens");
        let normalizer = take("normalizer");
        let pre = take("pre_tokenizer");
        let post = take("post_processor");
        let decoder = take("decoder");

        let mut tok = Self::from_model(model)?;
        tok.normalizer = parse_normalizer(normalizer.as_ref())?;
        tok.split = parse_pre_tokenizer(pre.as_ref())?;
        tok.set_added(added_json.as_ref())?;
        tok.template = tok.parse_post(post.as_ref())?;
        tok.decoder = parse_decoder(decoder.as_ref())?;
        Ok(tok)
    }

    fn from_model(model: Json) -> Result<Self, String> {
        if type_of(&model, "model")? != "BPE" {
            return Err("tokenizer.json: only a BPE model is supported".into());
        }
        for key in ["dropout", "continuing_subword_prefix", "end_of_word_suffix"] {
            match get(&model, key) {
                None | Some(Json::Null) => {}
                Some(_) => return Err(format!("tokenizer.json: model.{key} is not supported")),
            }
        }
        let fuse_unk = as_bool(get(&model, "fuse_unk"), "model.fuse_unk")?;
        let byte_fallback = as_bool(get(&model, "byte_fallback"), "model.byte_fallback")?;
        let ignore_merges = match get(&model, "ignore_merges") {
            None => false,
            j => as_bool(j, "model.ignore_merges")?,
        };
        let unk_text = match get(&model, "unk_token") {
            None | Some(Json::Null) => None,
            Some(Json::Str(s)) => Some(s.clone()),
            Some(_) => return Err("tokenizer.json: model.unk_token is not a string".into()),
        };

        let Json::Object(model_fields) = model else {
            return Err("tokenizer.json: model is not an object".into());
        };
        let mut vocab = None;
        let mut merges = None;
        for (k, v) in model_fields {
            match k.as_str() {
                "vocab" => vocab = Some(v),
                "merges" => merges = Some(v),
                _ => {}
            }
        }
        let Some(Json::Object(vocab)) = vocab else {
            return Err("tokenizer.json: model.vocab is not an object".into());
        };
        let Some(Json::Array(merges)) = merges else {
            return Err("tokenizer.json: model.merges is not an array".into());
        };

        let n = vocab.len();
        if n >= (1 << 31) {
            return Err("tokenizer.json: vocabulary too large".into());
        }
        let total: usize = vocab.iter().map(|(k, _)| k.len()).sum();
        if total >= u32::MAX as usize {
            return Err("tokenizer.json: vocabulary text too large".into());
        }
        let mut arena = Vec::with_capacity(total);
        let mut spans = vec![(NONE, NONE); n];
        for (text, idj) in &vocab {
            let id = as_u32(idj, "a vocabulary id")? as usize;
            if id >= n {
                return Err(format!(
                    "tokenizer.json: vocabulary id {id} is outside 0..{n} (ids must be dense)"
                ));
            }
            if spans[id].0 != NONE {
                return Err(format!("tokenizer.json: vocabulary id {id} is used twice"));
            }
            let start = arena.len() as u32;
            arena.extend_from_slice(text.as_bytes());
            spans[id] = (start, arena.len() as u32);
        }
        drop(vocab);

        let index = VocabIndex::build(&arena, &spans)?;
        let mut tok = Tokenizer {
            arena,
            spans,
            index,
            char_ids: vec![NONE; 0x11_0000],
            byte_ids: [NONE; 256],
            merges: MergeTable::with_capacity(merges.len()),
            byte_fallback,
            fuse_unk,
            ignore_merges,
            unk: None,
            normalizer: Vec::new(),
            split: None,
            added: Vec::new(),
            added_by_byte: vec![Vec::new(); 256],
            specials: Vec::new(),
            template: Vec::new(),
            decoder: Vec::new(),
            bos_token: None,
            placeholders: Vec::new(),
            prompts: Vec::new(),
            default_prompt: None,
        };

        for id in 0..n {
            let (s, e) = tok.spans[id];
            if let Ok(t) = std::str::from_utf8(&tok.arena[s as usize..e as usize]) {
                let mut it = t.chars();
                if let (Some(c), None) = (it.next(), it.next()) {
                    tok.char_ids[c as usize] = id as u32;
                }
            }
        }
        for b in 0..256usize {
            let name = format!("<{:#04X}>", b);
            if let Some(id) = tok.lookup(name.as_bytes()) {
                tok.byte_ids[b] = id;
            }
        }
        if let Some(u) = unk_text {
            tok.unk = Some(
                tok.lookup(u.as_bytes())
                    .ok_or_else(|| format!("tokenizer.json: unk_token {u:?} is not in the vocabulary"))?,
            );
        }

        let mut joined: Vec<u8> = Vec::new();
        for (rank, m) in merges.iter().enumerate() {
            let rank = rank as u32;
            let (a, b): (&str, &str) = match m {
                Json::Array(p) => match p.as_slice() {
                    [Json::Str(a), Json::Str(b)] => (a, b),
                    _ => return Err(format!("tokenizer.json: merge {rank} is not a pair of strings")),
                },
                Json::Str(s) => {
                    let mut parts = s.split(' ');
                    match (parts.next(), parts.next(), parts.next()) {
                        (Some(a), Some(b), None) => (a, b),
                        _ => {
                            return Err(format!(
                                "tokenizer.json: merge {rank} {s:?} is not two space-separated tokens"
                            ))
                        }
                    }
                }
                _ => return Err(format!("tokenizer.json: merge {rank} is neither a pair nor a string")),
            };
            let ida = tok
                .lookup(a.as_bytes())
                .ok_or_else(|| format!("tokenizer.json: merge {rank}: {a:?} is not in the vocabulary"))?;
            let idb = tok
                .lookup(b.as_bytes())
                .ok_or_else(|| format!("tokenizer.json: merge {rank}: {b:?} is not in the vocabulary"))?;
            joined.clear();
            joined.extend_from_slice(a.as_bytes());
            joined.extend_from_slice(b.as_bytes());
            let new = tok
                .lookup(&joined)
                .ok_or_else(|| format!("tokenizer.json: merge {rank}: {a:?} + {b:?} is not in the vocabulary"))?;
            tok.merges.insert(ida, idb, rank, new);
        }
        Ok(tok)
    }

    fn set_added(&mut self, added: Option<&Json>) -> Result<(), String> {
        let list = match added {
            None | Some(Json::Null) => return Ok(()),
            Some(Json::Array(l)) => l,
            Some(_) => return Err("tokenizer.json: added_tokens is not an array".into()),
        };
        for t in list {
            let content = as_str(
                get(t, "content").ok_or("tokenizer.json: an added token has no content")?,
                "an added token's content",
            )?;
            let id = as_u32(
                get(t, "id").ok_or("tokenizer.json: an added token has no id")?,
                "an added token's id",
            )?;
            let special = as_bool(get(t, "special"), "an added token's special")?;
            for flag in ["single_word", "lstrip", "rstrip", "normalized"] {
                if as_bool(get(t, flag), &format!("added token {content:?}: {flag}"))? {
                    return Err(format!(
                        "tokenizer.json: added token {content:?} has {flag}: true, which is not supported"
                    ));
                }
            }
            if content.is_empty() {
                return Err("tokenizer.json: an added token with empty content".into());
            }
            if self.lookup(content.as_bytes()) != Some(id) {
                return Err(format!(
                    "tokenizer.json: added token {content:?} (id {id}) is not the vocabulary's entry for that text"
                ));
            }
            if self.added.iter().any(|a| a.id == id) {
                return Err(format!("tokenizer.json: added token id {id} is listed twice"));
            }
            self.added.push(Added {
                bytes: content.as_bytes().to_vec(),
                id,
            });
            if special {
                self.specials.push(id);
            }
        }
        self.specials.sort_unstable();
        for (i, a) in self.added.iter().enumerate() {
            self.added_by_byte[a.bytes[0] as usize].push(i as u32);
        }
        let added = &self.added;
        for list in &mut self.added_by_byte {
            list.sort_by_key(|&i| Reverse(added[i as usize].bytes.len()));
        }
        Ok(())
    }

    fn parse_post(&self, post: Option<&Json>) -> Result<Vec<Template>, String> {
        let Some(post) = post.filter(|p| !matches!(p, Json::Null)) else {
            return Ok(vec![Template::Sequence]);
        };
        if type_of(post, "post_processor")? != "TemplateProcessing" {
            return Err("tokenizer.json: only a TemplateProcessing post_processor is supported".into());
        }
        let Some(Json::Array(single)) = get(post, "single") else {
            return Err("tokenizer.json: post_processor.single is not an array".into());
        };
        let specials = get(post, "special_tokens");
        let mut out = Vec::new();
        let mut sequences = 0;
        for item in single {
            if let Some(s) = get(item, "SpecialToken") {
                let id = as_str(
                    get(s, "id").ok_or("tokenizer.json: a SpecialToken has no id")?,
                    "a SpecialToken id",
                )?;
                let entry = specials
                    .and_then(|m| get(m, id))
                    .ok_or_else(|| format!("tokenizer.json: template token {id:?} is not in special_tokens"))?;
                let Some(Json::Array(ids)) = get(entry, "ids") else {
                    return Err(format!("tokenizer.json: special_tokens.{id}.ids is not an array"));
                };
                let ids = ids
                    .iter()
                    .map(|j| as_u32(j, "a template id"))
                    .collect::<Result<Vec<_>, _>>()?;
                if ids.iter().any(|&i| i as usize >= self.vocab_size()) {
                    return Err(format!(
                        "tokenizer.json: special_tokens.{id} has an id outside the vocabulary"
                    ));
                }
                out.push(Template::Special(ids));
            } else if let Some(s) = get(item, "Sequence") {
                if as_str(
                    get(s, "id").ok_or("tokenizer.json: a Sequence has no id")?,
                    "a Sequence id",
                )? != "A"
                {
                    return Err("tokenizer.json: the single template may only hold sequence A".into());
                }
                sequences += 1;
                out.push(Template::Sequence);
            } else {
                return Err("tokenizer.json: a template item is neither SpecialToken nor Sequence".into());
            }
        }
        if sequences != 1 {
            return Err("tokenizer.json: the single template must hold sequence A exactly once".into());
        }
        Ok(out)
    }

    fn is_special(&self, id: u32) -> bool {
        self.specials.binary_search(&id).is_ok()
    }

    // ---- encoding -------------------------------------------------------

    /// The ids `Tokenizer.encode(text, add_special_tokens).ids` of the
    /// `tokenizers` library gives, with no sentence-transformers behaviour on
    /// top: the added tokens in `text` are matched, the rest is normalized,
    /// pre-tokenized and merged, and the post-processor's template wraps it
    /// when `add_special_tokens`.
    ///
    /// # Panics
    ///
    /// On a `text` of [`MAX_TEXT_BYTES`] bytes or more (the merge list's
    /// positions are 32-bit); [`Self::encode_with_prompt`] refuses it as an
    /// error instead.
    pub fn encode_raw(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        assert!(
            text.len() < MAX_TEXT_BYTES,
            "encode_raw: a text of {} bytes is past the {MAX_TEXT_BYTES}-byte limit",
            text.len()
        );
        let mut body: Vec<u32> = Vec::with_capacity(text.len() / 3 + 4);
        let mut scratch = Scratch::default();
        let bytes = text.as_bytes();
        let (mut seg, mut i) = (0usize, 0usize);
        while i < bytes.len() {
            let cands = &self.added_by_byte[bytes[i] as usize];
            if !cands.is_empty() {
                if let Some(&a) = cands
                    .iter()
                    .find(|&&c| bytes[i..].starts_with(&self.added[c as usize].bytes))
                {
                    self.encode_text(&text[seg..i], &mut body, &mut scratch);
                    let a = &self.added[a as usize];
                    body.push(a.id);
                    i += a.bytes.len();
                    seg = i;
                    continue;
                }
            }
            i += 1;
        }
        self.encode_text(&text[seg..], &mut body, &mut scratch);
        if !add_special_tokens {
            return body;
        }
        let mut out = Vec::with_capacity(body.len() + 2);
        for item in &self.template {
            match item {
                Template::Special(ids) => out.extend_from_slice(ids),
                Template::Sequence => out.extend_from_slice(&body),
            }
        }
        out
    }

    /// Text between added tokens: normalizer, pre-tokenizer, BPE per piece.
    fn encode_text(&self, text: &str, out: &mut Vec<u32>, scratch: &mut Scratch) {
        if text.is_empty() {
            return;
        }
        let mut normalized: Cow<str> = Cow::Borrowed(text);
        for (from, to) in &self.normalizer {
            if normalized.contains(from.as_str()) {
                normalized = Cow::Owned(normalized.replace(from.as_str(), to));
            }
        }
        match &self.split {
            None => self.bpe(&normalized, out, scratch),
            Some(delim) => {
                for piece in split_merged_with_previous(&normalized, delim) {
                    self.bpe(piece, out, scratch);
                }
            }
        }
    }

    /// One pre-token through the BPE model (the library's `BPE::tokenize`).
    fn bpe(&self, piece: &str, out: &mut Vec<u32>, scratch: &mut Scratch) {
        if piece.is_empty() {
            return;
        }
        if self.ignore_merges {
            if let Some(id) = self.lookup(piece.as_bytes()) {
                out.push(id);
                return;
            }
        }
        let syms = &mut scratch.symbols;
        syms.clear();
        let push = |syms: &mut Vec<Symbol>, c: u32, len: usize| {
            syms.push(Symbol {
                c,
                prev: syms.len() as i32 - 1,
                next: -1,
                len: len as u32,
            });
        };
        // (id, byte length) of an unknown-character run not yet written,
        // exactly as `merge_word` carries it.
        let mut unk: Option<(u32, usize)> = None;
        for ch in piece.chars() {
            let len = ch.len_utf8();
            let id = self.char_ids[ch as usize];
            if id != NONE {
                if let Some((u, l)) = unk.take() {
                    push(syms, u, l);
                }
                push(syms, id, len);
                continue;
            }
            if self.byte_fallback {
                let mut buf = [0u8; 4];
                let enc = ch.encode_utf8(&mut buf).as_bytes();
                if enc.iter().all(|&b| self.byte_ids[b as usize] != NONE) {
                    for &b in enc {
                        push(syms, self.byte_ids[b as usize], 1);
                    }
                    continue;
                }
            }
            if let Some(unk_id) = self.unk {
                unk = match (unk, self.fuse_unk) {
                    (Some((u, l)), true) => Some((u, l + len)),
                    (Some((u, l)), false) => {
                        push(syms, u, l);
                        Some((unk_id, len))
                    }
                    (None, _) => Some((unk_id, len)),
                };
            }
        }
        if let Some((u, l)) = unk {
            push(syms, u, l);
        }
        let n = syms.len();
        if n == 0 {
            return;
        }
        for (i, s) in syms.iter_mut().enumerate() {
            s.next = if i + 1 < n { i as i32 + 1 } else { -1 };
        }

        // The library's `Word::merge_all`: a min-heap on (rank, position).
        let heap = &mut scratch.heap;
        heap.clear();
        for i in 0..n - 1 {
            if let Some((rank, new)) = self.merges.get(syms[i].c, syms[i + 1].c) {
                heap.push(Reverse((rank, i as u32, new)));
            }
        }
        while let Some(Reverse((_, pos, new))) = heap.pop() {
            let pos = pos as usize;
            if syms[pos].len == 0 || syms[pos].next == -1 {
                continue;
            }
            let next_pos = syms[pos].next as usize;
            let right = syms[next_pos];
            // An entry whose pair has since changed is stale.
            match self.merges.get(syms[pos].c, right.c) {
                Some((_, n2)) if n2 == new => {}
                _ => continue,
            }
            syms[pos].c = new;
            syms[pos].len += right.len;
            syms[pos].next = right.next;
            syms[next_pos].len = 0;
            if right.next > -1 && (right.next as usize) < n {
                syms[right.next as usize].prev = pos as i32;
            }
            let cur = syms[pos];
            if cur.prev >= 0 {
                let prev = syms[cur.prev as usize];
                if let Some((rank, nn)) = self.merges.get(prev.c, cur.c) {
                    heap.push(Reverse((rank, cur.prev as u32, nn)));
                }
            }
            if cur.next >= 0 && (cur.next as usize) < n {
                let next = syms[cur.next as usize];
                if let Some((rank, nn)) = self.merges.get(cur.c, next.c) {
                    heap.push(Reverse((rank, pos as u32, nn)));
                }
            }
        }
        out.extend(syms.iter().filter(|s| s.len != 0).map(|s| s.c));
    }

    /// The text the tokenizer sees for `prompt_name` and `text`: the prompt
    /// written in front of the text. An unknown name is an error.
    fn render(&self, prompt_name: Option<&str>, text: &str) -> Result<String, String> {
        let Some(name) = prompt_name else {
            return Ok(text.to_string());
        };
        let prompt = self.prompt(name).ok_or_else(|| {
            let names: Vec<&str> = self.prompt_names().collect();
            format!("no prompt named {name:?} (the snapshot has {names:?})")
        })?;
        let mut s = String::with_capacity(prompt.len() + text.len());
        s.push_str(prompt);
        s.push_str(text);
        Ok(s)
    }

    /// The ids sentence-transformers feeds the model for one text, as a batch
    /// of one: see the module docs. `prompt_name` names one of
    /// [`Self::prompt_names`] (`encode(prompt_name=...)`); `None` is no prompt.
    pub fn encode_with_prompt(&self, prompt_name: Option<&str>, text: &str) -> Result<Vec<u32>, String> {
        let rendered = self.render(prompt_name, text)?;
        if rendered.len() >= MAX_TEXT_BYTES {
            return Err(format!(
                "a text of {} bytes (with its prompt) is past the {MAX_TEXT_BYTES}-byte limit",
                rendered.len()
            ));
        }
        for p in &self.placeholders {
            if rendered.contains(p.as_str()) {
                return Err(format!(
                    "the text holds the placeholder {p:?}, which sentence-transformers refuses without media to fill it"
                ));
            }
        }
        let skip = self.bos_token.as_deref().is_some_and(|b| rendered.starts_with(b));
        Ok(self.encode_raw(&rendered, !skip))
    }

    /// [`Self::encode_with_prompt`] with no prompt.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode_with_prompt(None, text)
    }

    /// Each of `texts` as [`Self::encode_with_prompt`] encodes it, in order.
    /// The first error (naming the text's position) aborts the batch.
    ///
    /// A batch large enough to pay for it is spread over threads: contiguous
    /// chunks balanced by byte length, joined in order, so the ids and the
    /// error are those of a single thread. The thread count is the
    /// `TESSL_TOK_THREADS` environment variable (1 to 64), by default the
    /// machine's cores up to 32; a bad value is an error. An empty batch, one
    /// text, or under 16 KiB of text runs on the calling thread and starts
    /// none. See [`Self::encode_batch_with_threads`].
    pub fn encode_batch_with_prompt(&self, prompt_name: Option<&str>, texts: &[&str]) -> Result<Vec<Vec<u32>>, String> {
        if texts.len() < 2 {
            return self.encode_batch_with_threads(prompt_name, texts, 1);
        }
        let threads = crate::par_batch::configured_threads()?;
        self.encode_batch_with_threads(prompt_name, texts, threads)
    }

    /// [`Self::encode_batch_with_prompt`] over at most `threads` threads
    /// (clamped to 1 to 64), whatever the environment says. The result does
    /// not depend on `threads`.
    pub fn encode_batch_with_threads(
        &self,
        prompt_name: Option<&str>,
        texts: &[&str],
        threads: usize,
    ) -> Result<Vec<Vec<u32>>, String> {
        crate::par_batch::map_ordered(texts, threads, &|i, t| {
            self.encode_with_prompt(prompt_name, t)
                .map_err(|e| format!("text {i}: {e}"))
        })
    }

    /// [`Self::encode_batch_with_prompt`] with no prompt.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, String> {
        self.encode_batch_with_prompt(None, texts)
    }

    // ---- decoding -------------------------------------------------------

    /// `Tokenizer.decode(ids)` of the `tokenizers` library: special tokens
    /// are skipped, as that method's default. An id with no token is skipped
    /// too, as the library does.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.decode_with(ids, true)
    }

    /// [`Self::decode`] with `skip_special_tokens` chosen. (The library's
    /// method defaults to `true`; transformers' `decode` defaults to `false`.)
    pub fn decode_with(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        let mut tokens: Vec<String> = ids
            .iter()
            .filter(|&&id| !(skip_special_tokens && self.is_special(id)))
            .filter_map(|&id| self.id_to_token(id))
            .map(str::to_string)
            .collect();
        for step in &self.decoder {
            tokens = match step {
                DecodeStep::Replace { from, to } => tokens
                    .into_iter()
                    .map(|t| {
                        if t.contains(from.as_str()) {
                            t.replace(from.as_str(), to)
                        } else {
                            t
                        }
                    })
                    .collect(),
                DecodeStep::ByteFallback => byte_fallback(tokens),
                DecodeStep::Fuse => vec![tokens.concat()],
            };
        }
        tokens.concat()
    }
}

/// The working buffers of one encode, reused across its pieces.
#[derive(Default)]
struct Scratch {
    symbols: Vec<Symbol>,
    heap: BinaryHeap<Reverse<(u32, u32, u32)>>,
}

/// The library's `Split` with a literal pattern in `MergedWithPrevious` mode:
/// a delimiter joins the piece before it, and a delimiter right after another
/// delimiter is a piece of its own.
fn split_merged_with_previous<'a>(text: &'a str, delim: &str) -> Vec<&'a str> {
    // The delimiters and the gaps between them, in order: (start, end, is_delim).
    let mut segments: Vec<(usize, usize, bool)> = Vec::new();
    let mut prev = 0usize;
    for (at, _) in text.match_indices(delim) {
        if at > prev {
            segments.push((prev, at, false));
        }
        segments.push((at, at + delim.len(), true));
        prev = at + delim.len();
    }
    if prev < text.len() {
        segments.push((prev, text.len(), false));
    }
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    let mut previous_match = false;
    for (start, end, is_match) in segments {
        match pieces.last_mut() {
            Some(last) if is_match && !previous_match => last.1 = end,
            _ => pieces.push((start, end)),
        }
        previous_match = is_match;
    }
    pieces.into_iter().map(|(s, e)| &text[s..e]).collect()
}

/// The library's `ByteFallback` decoder: a run of `<0xHH>` tokens is one
/// UTF-8 string, or one `U+FFFD` per byte when the run is not valid UTF-8.
fn byte_fallback(tokens: Vec<String>) -> Vec<String> {
    fn flush(run: &mut Vec<u8>, out: &mut Vec<String>) {
        if run.is_empty() {
            return;
        }
        match std::str::from_utf8(run) {
            Ok(s) => out.push(s.to_string()),
            Err(_) => out.extend(std::iter::repeat_n("\u{fffd}".to_string(), run.len())),
        }
        run.clear();
    }
    let mut out = Vec::with_capacity(tokens.len());
    let mut run: Vec<u8> = Vec::new();
    for t in tokens {
        let byte = t
            .strip_prefix("<0x")
            .and_then(|r| r.strip_suffix('>'))
            .filter(|h| h.len() == 2 && h.bytes().all(|b| b.is_ascii_hexdigit()))
            .and_then(|h| u8::from_str_radix(h, 16).ok());
        match byte {
            Some(b) => run.push(b),
            None => {
                flush(&mut run, &mut out);
                out.push(t);
            }
        }
    }
    flush(&mut run, &mut out);
    out
}

fn parse_normalizer(j: Option<&Json>) -> Result<Vec<(String, String)>, String> {
    let Some(j) = j.filter(|j| !matches!(j, Json::Null)) else {
        return Ok(Vec::new());
    };
    match type_of(j, "normalizer")? {
        "Replace" => Ok(vec![parse_replace(j, "the Replace normalizer")?]),
        "Sequence" => {
            let Some(Json::Array(items)) = get(j, "normalizers") else {
                return Err("tokenizer.json: normalizer.normalizers is not an array".into());
            };
            let mut out = Vec::new();
            for it in items {
                if type_of(it, "a normalizer")? != "Replace" {
                    return Err(format!(
                        "tokenizer.json: normalizer {:?} is not supported",
                        type_of(it, "")?
                    ));
                }
                out.push(parse_replace(it, "a Replace normalizer")?);
            }
            Ok(out)
        }
        other => Err(format!("tokenizer.json: normalizer {other:?} is not supported")),
    }
}

fn parse_pre_tokenizer(j: Option<&Json>) -> Result<Option<String>, String> {
    let Some(j) = j.filter(|j| !matches!(j, Json::Null)) else {
        return Ok(None);
    };
    match type_of(j, "pre_tokenizer")? {
        "Split" => {
            let delim = string_pattern(get(j, "pattern"), "the Split pre-tokenizer")?;
            match (get(j, "behavior"), get(j, "invert")) {
                (Some(Json::Str(b)), Some(Json::Bool(false))) if b == "MergedWithPrevious" => Ok(Some(delim)),
                _ => Err(
                    "tokenizer.json: the Split pre-tokenizer is supported only as MergedWithPrevious, invert false"
                        .into(),
                ),
            }
        }
        other => Err(format!("tokenizer.json: pre_tokenizer {other:?} is not supported")),
    }
}

fn parse_decoder(j: Option<&Json>) -> Result<Vec<DecodeStep>, String> {
    fn step(j: &Json) -> Result<DecodeStep, String> {
        match type_of(j, "a decoder")? {
            "Replace" => {
                let (from, to) = parse_replace(j, "the Replace decoder")?;
                Ok(DecodeStep::Replace { from, to })
            }
            "ByteFallback" => Ok(DecodeStep::ByteFallback),
            "Fuse" => Ok(DecodeStep::Fuse),
            other => Err(format!("tokenizer.json: decoder {other:?} is not supported")),
        }
    }
    let Some(j) = j.filter(|j| !matches!(j, Json::Null)) else {
        return Err(
            "tokenizer.json: a null decoder is not supported (the library would join tokens with spaces)".into(),
        );
    };
    if type_of(j, "decoder")? == "Sequence" {
        let Some(Json::Array(items)) = get(j, "decoders") else {
            return Err("tokenizer.json: decoder.decoders is not an array".into());
        };
        items.iter().map(step).collect()
    } else {
        Ok(vec![step(j)?])
    }
}

fn read_bounded(path: &Path, cap: u64) -> Result<String, String> {
    let (file, len) = crate::plain::open_regular_file(path)?;
    if len > cap {
        return Err(format!("{}: {len} bytes, above the {cap}-byte cap", path.display()));
    }
    let mut text = String::with_capacity(len as usize);
    file.take(cap)
        .read_to_string(&mut text)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_merge_table_finds_every_pair_across_collisions_and_overwrites_a_repeat() {
        let mut t = MergeTable::with_capacity(5000);
        for i in 0..5000u32 {
            t.insert(i.wrapping_mul(2_654_435_761) >> 12, i, i, i + 1);
        }
        for i in 0..5000u32 {
            assert_eq!(t.get(i.wrapping_mul(2_654_435_761) >> 12, i), Some((i, i + 1)));
        }
        assert_eq!(t.get(7, 7_000_000), None);
        t.insert(1, 2, 10, 20);
        t.insert(1, 2, 11, 21);
        assert_eq!(t.get(1, 2), Some((11, 21)), "the later line wins");
        // (a, b) and (b, a) are different pairs.
        assert_eq!(t.get(2, 1), None);
    }

    #[test]
    fn a_split_merges_each_delimiter_into_the_piece_before_it() {
        for (text, want) in [
            ("", vec![]),
            (" ", vec![" "]),
            ("ab", vec!["ab"]),
            ("a b c", vec!["a ", "b ", "c"]),
            ("a  b", vec!["a ", " ", "b"]),
            ("  a", vec![" ", " ", "a"]),
            (" a ", vec![" ", "a "]),
            ("a   ", vec!["a ", " ", " "]),
        ] {
            assert_eq!(split_merged_with_previous(text, " "), want, "{text:?}");
        }
        assert_eq!(split_merged_with_previous("xabyabz", "ab"), ["xab", "yab", "z"]);
    }

    #[test]
    fn a_byte_run_is_one_string_or_one_replacement_per_byte() {
        let t = |s: &[&str]| byte_fallback(s.iter().map(|x| (*x).to_string()).collect::<Vec<_>>());
        assert_eq!(t(&["<0xE6>", "<0x97>", "<0xA5>"]), ["日"]);
        assert_eq!(t(&["<0xE6>", "<0x97>"]), ["\u{fffd}", "\u{fffd}"]);
        assert_eq!(t(&["a", "<0xFF>", "b"]), ["a", "\u{fffd}", "b"]);
        assert_eq!(t(&["<0xC3>", "<0xA9>", "<0xC3>"]), ["\u{fffd}"; 3]);
        // Not a byte token: wrong length, not hex, upper-case x.
        assert_eq!(
            t(&["<0x4>", "<0xZZ>", "<0X41>", "<0x41"]),
            ["<0x4>", "<0xZZ>", "<0X41>", "<0x41"]
        );
        assert_eq!(t(&["<0x41>", "<0x42>"]), ["AB"]);
    }

    #[test]
    fn the_index_refuses_two_entries_with_the_same_text() {
        let arena = b"abab".to_vec();
        assert!(VocabIndex::build(&arena, &[(0, 2), (2, 4)])
            .err()
            .unwrap()
            .contains("same text"));
        let ok = VocabIndex::build(&arena, &[(0, 2), (1, 3)]).ok().unwrap();
        assert_eq!(ok.slots.iter().filter(|&&s| s != 0).count(), 2);
    }
}
