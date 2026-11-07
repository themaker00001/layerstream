//! Qwen2 byte-level BPE tokenizer, written from scratch.
//!
//! Encoding: split the text on special tokens, split the rest into "pre-tokens" with the
//! Qwen2 pattern (hand-coded below, since there is no regex engine), map each pre-token's
//! bytes to printable characters (the GPT-2 byte table), then apply BPE merges by rank.
//!
//! The vocabulary is stored compactly (one byte arena, a flat hash table and a sorted merge
//! list) because the obvious `HashMap<String, u32>` plus a JSON tree cost more memory than the
//! model layers being streamed.
//!
//! Known approximations: Unicode letter and number classes use Rust's `char::is_alphabetic`
//! and `char::is_numeric` instead of the exact `\p{L}` and `\p{N}` tables, and text is not
//! NFC-normalised first. For ordinary text these agree with the reference tokenizer; the test
//! suite compares them on a varied set of strings.

use crate::json::{for_each_int_entry, Json};
use std::collections::HashMap;
use std::path::Path;

/// The GPT-2 byte-to-unicode table: printable bytes map to themselves, the rest to
/// code points starting at 256.
fn byte_table() -> [char; 256] {
    let mut keep: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut mapped = keep.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !keep.contains(&b) {
            keep.push(b);
            mapped.push(256 + n);
            n += 1;
        }
    }
    let mut table = ['\0'; 256];
    for (b, c) in keep.iter().zip(mapped.iter()) {
        table[*b as usize] = char::from_u32(*c).unwrap();
    }
    table
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[derive(Clone, Copy)]
struct MergeEntry {
    key: u64, // (left << 32) | right
    rank: u32,
    merged: u32,
}

pub struct Tokenizer {
    /// All token strings back to back (vocabulary entries first, then added tokens).
    arena: Vec<u8>,
    /// `spans[id]` is the token's byte range in the arena (empty for unused ids).
    spans: Vec<(u32, u32)>,
    /// Open-addressing table of `id + 1` keyed by the hash of a vocabulary token's bytes.
    table: Vec<u32>,
    merges: Vec<MergeEntry>,
    byte_ids: [u32; 256],
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    /// Added tokens (specials and others), longest first, matched literally in text.
    added: Vec<(String, u32)>,
    special_ids: Vec<bool>,
}

impl Tokenizer {
    pub fn load(dir: &Path) -> Result<Tokenizer, String> {
        let read = |name: &str| std::fs::read_to_string(dir.join(name)).map_err(|e| format!("{name}: {e}"));

        // Vocabulary: stream the file straight into the arena.
        let mut arena: Vec<u8> = Vec::with_capacity(2 << 20);
        let mut spans: Vec<(u32, u32)> = Vec::new();
        let mut n_vocab = 0usize;
        {
            let text = read("vocab.json")?;
            for_each_int_entry(&text, &mut |tok, id| {
                let id = id as usize;
                if id >= spans.len() {
                    spans.resize(id + 1, (0, 0));
                }
                let start = arena.len() as u32;
                arena.extend_from_slice(tok.as_bytes());
                spans[id] = (start, arena.len() as u32);
                n_vocab += 1;
                Ok(())
            })?;
        }

        // Added tokens.
        let cfg = Json::parse(&read("tokenizer_config.json")?)?;
        let mut added: Vec<(String, u32)> = Vec::new();
        let mut specials: Vec<u32> = Vec::new();
        if let Some(items) = cfg.get("added_tokens_decoder").and_then(Json::as_obj) {
            for (id, v) in items {
                let id: u32 = id.parse().map_err(|_| "bad added token id")?;
                let content = v.get("content").and_then(Json::as_str).ok_or("added token without content")?;
                added.push((content.to_string(), id));
                if v.get("special").and_then(Json::as_bool).unwrap_or(false) {
                    specials.push(id);
                }
            }
        }
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        let mut special_ids = vec![false; spans.len()];
        for (content, id) in &added {
            let id = *id as usize;
            if id >= spans.len() {
                spans.resize(id + 1, (0, 0));
                special_ids.resize(id + 1, false);
            }
            let start = arena.len() as u32;
            arena.extend_from_slice(content.as_bytes());
            spans[id] = (start, arena.len() as u32);
        }
        for id in specials {
            special_ids[id as usize] = true;
        }
        let vocab_ids = n_vocab;
        let _ = vocab_ids;

        // Hash table over the vocabulary (not the added tokens: those never come from BPE).
        let added_ids: std::collections::HashSet<u32> = added.iter().map(|(_, id)| *id).collect();
        let size = (n_vocab * 2).next_power_of_two().max(16);
        let mut table = vec![0u32; size];
        for (id, &(s, e)) in spans.iter().enumerate() {
            if e > s && !added_ids.contains(&(id as u32)) {
                let mut slot = fnv1a(&arena[s as usize..e as usize]) as usize & (size - 1);
                while table[slot] != 0 {
                    slot = (slot + 1) & (size - 1);
                }
                table[slot] = id as u32 + 1;
            }
        }

        let mut tok = Tokenizer {
            arena,
            spans,
            table,
            merges: Vec::new(),
            byte_ids: [0; 256],
            byte_to_char: byte_table(),
            char_to_byte: HashMap::new(),
            added,
            special_ids,
        };
        tok.char_to_byte = tok.byte_to_char.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        for b in 0..256 {
            let mut buf = [0u8; 4];
            let s = tok.byte_to_char[b].encode_utf8(&mut buf);
            tok.byte_ids[b] = tok.lookup(s.as_bytes()).ok_or("a single-byte token is missing from the vocabulary")?;
        }

        // Merges.
        let merges_text = read("merges.txt")?;
        let mut merges: Vec<MergeEntry> = Vec::with_capacity(merges_text.len() / 12);
        let mut joined: Vec<u8> = Vec::new();
        for (rank, line) in merges_text.lines().filter(|l| !l.starts_with("#version") && !l.is_empty()).enumerate() {
            let (a, b) = line.split_once(' ').ok_or_else(|| format!("bad merge line {line:?}"))?;
            let ia = tok.lookup(a.as_bytes()).ok_or("merge token missing from vocab")?;
            let ib = tok.lookup(b.as_bytes()).ok_or("merge token missing from vocab")?;
            joined.clear();
            joined.extend_from_slice(a.as_bytes());
            joined.extend_from_slice(b.as_bytes());
            let merged = tok.lookup(&joined).ok_or("merged token missing from vocab")?;
            merges.push(MergeEntry { key: ((ia as u64) << 32) | ib as u64, rank: rank as u32, merged });
        }
        drop(merges_text);
        merges.sort_by_key(|m| (m.key, m.rank));
        merges.dedup_by_key(|m| m.key); // keep the lowest rank of any duplicate pair
        merges.shrink_to_fit();
        tok.merges = merges;
        Ok(tok)
    }

    /// Id of a vocabulary token given its (byte-level) string.
    fn lookup(&self, key: &[u8]) -> Option<u32> {
        let mask = self.table.len() - 1;
        let mut slot = fnv1a(key) as usize & mask;
        loop {
            let v = self.table[slot];
            if v == 0 {
                return None;
            }
            let (s, e) = self.spans[(v - 1) as usize];
            if &self.arena[s as usize..e as usize] == key {
                return Some(v - 1);
            }
            slot = (slot + 1) & mask;
        }
    }

    fn merge_for(&self, left: u32, right: u32) -> Option<(u32, u32)> {
        let key = ((left as u64) << 32) | right as u64;
        self.merges.binary_search_by_key(&key, |m| m.key).ok().map(|i| (self.merges[i].rank, self.merges[i].merged))
    }

    pub fn vocab_size(&self) -> usize {
        self.spans.len()
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.added.iter().find(|(t, _)| t == token).map(|(_, id)| *id).or_else(|| self.lookup(token.as_bytes()))
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special_ids.get(id as usize).copied().unwrap_or(false)
    }

    /// Approximate heap use of the tokenizer, in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.arena.capacity() + self.spans.capacity() * 8 + self.table.capacity() * 4 + self.merges.capacity() * 16
    }

    /// Encode text, recognising added tokens such as `<|im_start|>` literally.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // Find the earliest added token occurrence (longest wins on ties).
            let mut best: Option<(usize, &str, u32)> = None;
            for (tok, id) in &self.added {
                if let Some(pos) = rest.find(tok.as_str()) {
                    if best.is_none_or(|(p, _, _)| pos < p) {
                        best = Some((pos, tok, *id));
                    }
                }
            }
            match best {
                Some((pos, tok, id)) => {
                    self.encode_ordinary(&rest[..pos], &mut out);
                    out.push(id);
                    rest = &rest[pos + tok.len()..];
                }
                None => {
                    self.encode_ordinary(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_ordinary(&self, text: &str, out: &mut Vec<u32>) {
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let end = pretoken_end(&chars, i);
            let piece: String = chars[i..end].iter().collect();
            self.bpe(piece.as_bytes(), out);
            i = end;
        }
    }

    fn bpe(&self, bytes: &[u8], out: &mut Vec<u32>) {
        let mut ids: Vec<u32> = bytes.iter().map(|b| self.byte_ids[*b as usize]).collect();
        while ids.len() > 1 {
            let mut best: Option<(u32, u32, u32, u32)> = None; // rank, left, right, merged
            for w in ids.windows(2) {
                if let Some((rank, merged)) = self.merge_for(w[0], w[1]) {
                    if best.is_none_or(|(r, _, _, _)| rank < r) {
                        best = Some((rank, w[0], w[1], merged));
                    }
                }
            }
            let Some((_, left, right, merged)) = best else { break };
            let mut next = Vec::with_capacity(ids.len());
            let mut k = 0;
            while k < ids.len() {
                if k + 1 < ids.len() && ids[k] == left && ids[k + 1] == right {
                    next.push(merged);
                    k += 2;
                } else {
                    next.push(ids[k]);
                    k += 1;
                }
            }
            ids = next;
        }
        out.extend(ids);
    }

    fn token_str(&self, id: u32) -> &str {
        match self.spans.get(id as usize) {
            Some(&(s, e)) => std::str::from_utf8(&self.arena[s as usize..e as usize]).unwrap_or(""),
            None => "",
        }
    }

    /// The bytes a token stands for. Special tokens yield no bytes unless `show_special`.
    pub fn token_bytes(&self, id: u32, show_special: bool) -> Vec<u8> {
        let tok = self.token_str(id);
        if tok.is_empty() {
            return Vec::new();
        }
        if self.is_special(id) {
            return if show_special { tok.as_bytes().to_vec() } else { Vec::new() };
        }
        if self.added.iter().any(|(_, a)| *a == id) {
            return tok.as_bytes().to_vec();
        }
        tok.chars().filter_map(|c| self.char_to_byte.get(&c).copied()).collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|id| self.token_bytes(*id, false)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Incremental UTF-8 decoder: feed token bytes, get back the text that is complete so far.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(e) => {
                let valid = e.valid_up_to();
                let out = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                if let Some(bad) = e.error_len() {
                    // Genuinely invalid bytes: emit a replacement and drop them.
                    let mut s = out;
                    s.push('\u{FFFD}');
                    self.pending.drain(..valid + bad);
                    s.push_str(&self.push(&[]));
                    s
                } else {
                    self.pending.drain(..valid);
                    out
                }
            }
        }
    }

    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

// ---------------------------------------------------------------------------
// Pre-tokenizer: the Qwen2 pattern, hand-coded
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   | [^\r\n\p{L}\p{N}]?\p{L}+
//   | \p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
//
// Alternatives are tried in order at each position; the first that matches wins.
// ---------------------------------------------------------------------------

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

fn is_number(c: char) -> bool {
    c.is_numeric()
}

fn is_nl(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// End (exclusive) of the pre-token that starts at `i`. Always advances by at least one.
fn pretoken_end(c: &[char], i: usize) -> usize {
    let n = c.len();

    // 1. contractions, case-insensitive
    if c[i] == '\'' {
        let lower = |k: usize| c.get(k).map(|ch| ch.to_ascii_lowercase());
        let two = (lower(i + 1), lower(i + 2));
        if matches!(two, (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l'))) {
            return i + 3;
        }
        if matches!(lower(i + 1), Some('s') | Some('t') | Some('m') | Some('d')) {
            return i + 2;
        }
    }

    // 2. optional single non-letter/number/newline, then letters
    {
        let mut j = i;
        if !is_nl(c[j]) && !is_letter(c[j]) && !is_number(c[j]) {
            j += 1;
        }
        if j < n && is_letter(c[j]) {
            while j < n && is_letter(c[j]) {
                j += 1;
            }
            return j;
        }
    }

    // 3. a single number
    if is_number(c[i]) {
        return i + 1;
    }

    // 4. optional space, then punctuation/symbols, then trailing newlines
    {
        let mut j = i;
        if c[j] == ' ' {
            j += 1;
        }
        let symbol = |ch: char| !ch.is_whitespace() && !is_letter(ch) && !is_number(ch);
        if j < n && symbol(c[j]) {
            while j < n && symbol(c[j]) {
                j += 1;
            }
            while j < n && is_nl(c[j]) {
                j += 1;
            }
            return j;
        }
    }

    // Whitespace alternatives (5-7) need a whitespace run.
    if c[i].is_whitespace() {
        let mut m = i;
        while m < n && c[m].is_whitespace() {
            m += 1;
        }
        // 5. whitespace ending in the last newline of the run
        if let Some(k) = (i..m).rev().find(|&k| is_nl(c[k])) {
            return k + 1;
        }
        // 6. whitespace not followed by a non-space: give back the last char unless at the end
        if m == n {
            return m;
        }
        if m - i >= 2 {
            return m - 1;
        }
        // 7. a single whitespace character
        return m;
    }

    i + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(s: &str) -> Vec<String> {
        let c: Vec<char> = s.chars().collect();
        let mut out = vec![];
        let mut i = 0;
        while i < c.len() {
            let e = pretoken_end(&c, i);
            assert!(e > i, "no progress at {i} in {s:?}");
            out.push(c[i..e].iter().collect());
            i = e;
        }
        out
    }

    #[test]
    fn splits_like_the_reference_pattern() {
        assert_eq!(pieces("Hello, world!"), ["Hello", ",", " world", "!"]);
        assert_eq!(pieces("I'm sure it's 2024"), ["I", "'m", " sure", " it", "'s", " ", "2", "0", "2", "4"]);
        assert_eq!(pieces("a  b"), ["a", " ", " b"]);
        assert_eq!(pieces("end.\n\nNext"), ["end", ".\n\n", "Next"]);
        assert_eq!(pieces("x \n y"), ["x", " \n", " y"]);
        assert_eq!(pieces("trailing   "), ["trailing", "   "]);
        assert_eq!(pieces("!!word"), ["!!", "word"]);
        assert_eq!(pieces("!word"), ["!word"]);
    }

    #[test]
    fn stream_decoder_waits_for_complete_characters() {
        let mut d = StreamDecoder::default();
        let smile = "😀".as_bytes();
        assert_eq!(d.push(&smile[..2]), "");
        assert_eq!(d.push(&smile[2..]), "😀");
        assert_eq!(d.push(b"ab"), "ab");
        assert_eq!(d.push(&[0xFF]), "\u{FFFD}");
        assert_eq!(d.finish(), "");
    }

    #[test]
    fn byte_table_is_a_bijection_onto_printable_chars() {
        let t = byte_table();
        let mut seen = std::collections::HashSet::new();
        for c in t {
            assert!(seen.insert(c));
        }
        assert_eq!(t[b'A' as usize], 'A');
        assert_eq!(t[b' ' as usize], 'Ġ');
        assert_eq!(t[b'\n' as usize], 'Ċ');
    }

    #[test]
    fn hash_table_finds_every_token_and_only_those() {
        let dir = std::env::temp_dir().join(format!("ls-tok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A tiny vocabulary: all 256 byte tokens plus two merges.
        let t = byte_table();
        let mut vocab: Vec<(String, usize)> = (0..256).map(|b| (t[b].to_string(), b)).collect();
        let (h, i) = (t[b'h' as usize], t[b'i' as usize]);
        vocab.push((format!("{h}{i}"), 256));
        vocab.push((format!("{h}{i}{h}{i}"), 257));
        let body: Vec<String> = vocab.iter().map(|(s, id)| format!("{}:{id}", crate::json::Json::Str(s.clone()).to_text())).collect();
        std::fs::write(dir.join("vocab.json"), format!("{{{}}}", body.join(","))).unwrap();
        std::fs::write(dir.join("merges.txt"), format!("#version: 0.2\n{h} {i}\n{h}{i} {h}{i}\n")).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), r#"{"added_tokens_decoder":{"300":{"content":"<|x|>","special":true}}}"#).unwrap();
        let tok = Tokenizer::load(&dir).unwrap();
        assert_eq!(tok.encode("hihi"), vec![257]);
        assert_eq!(tok.encode("hihihi"), vec![257, 256]);
        assert_eq!(tok.encode("hi"), vec![256]);
        assert_eq!(tok.encode("h"), vec![b'h' as u32]);
        assert_eq!(tok.encode("a<|x|>hi"), vec![b'a' as u32, 300, 256]);
        assert_eq!(tok.decode(&tok.encode("hihihi")), "hihihi");
        assert_eq!(tok.token_id("<|x|>"), Some(300));
        assert_eq!(tok.token_id("nonexistent token"), None);
        assert!(tok.is_special(300) && !tok.is_special(256));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
