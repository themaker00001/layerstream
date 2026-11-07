//! Encryption at rest for the model file: ChaCha20-Poly1305 (RFC 8439), applied per chunk so
//! a layer can be read and decrypted on its own, and decrypted in parallel.
//!
//! The cipher is a from-scratch implementation that keeps the crate dependency-free. It is
//! tested against the RFC vectors and against Python's `cryptography` package, but it has not
//! been audited and is not hardened beyond avoiding secret-dependent branches and indices. For
//! anything real, use a vetted library.
//!
//! What sealing buys: a copy of the file alone is useless without the key, and any change to
//! the file is detected. What it does not buy: the key and each layer are in memory while the
//! model runs, so anyone who can read this process's memory still gets them.
//!
//! File layout (integers little-endian):
//! ```text
//! 0   4   magic "LSS1"
//! 4   4   format version (1)
//! 8   8   nonce prefix (random per file)
//! 16  4   header length H
//! 20  H   header (JSON: model config, layer layout, chunk sizes)
//! ..  16 * n_chunks   authentication tags, one per chunk
//! ..      ciphertext, the same length as the plaintext
//! ```
//! The plaintext is, in order: every layer block, the embedding table, the final norm. Layer
//! blocks are cut into `layer_chunk`-byte chunks and the table into `embed_chunk_rows`-row
//! chunks; each chunk is its own AEAD message. Chunk `i` uses nonce `prefix || u32(i)` and
//! authenticates the whole prefix of the file up to the end of the header plus `u32(i)`, so
//! editing the header, moving chunks or flipping any bit is detected.

use crate::json::Json;
use crate::source::{FileSource, IoStats, Source, Throttle};
use crate::sys;
use crate::weights::{Config, LayerLayout};
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

pub const KEY_LEN: usize = 32;
pub const NONCE_LEN: usize = 12;
pub const TAG_LEN: usize = 16;

// ---------------------------------------------------------------------------
// ChaCha20, four blocks at a time
// ---------------------------------------------------------------------------

const SIGMA: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn wipe_words(w: &mut [u32]) {
    for x in w.iter_mut() {
        // SAFETY: `x` is a valid exclusive reference.
        unsafe { std::ptr::write_volatile(x, 0) };
    }
    std::sync::atomic::compiler_fence(Ordering::SeqCst);
}

fn init_state(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN]) -> [u32; 16] {
    let mut s = [0u32; 16];
    s[..4].copy_from_slice(&SIGMA);
    for i in 0..8 {
        s[4 + i] = le32(key, 4 * i);
    }
    for i in 0..3 {
        s[13 + i] = le32(nonce, 4 * i);
    }
    s
}

#[inline(always)]
fn add4(x: &mut [u32; 4], y: &[u32; 4]) {
    for (p, q) in x.iter_mut().zip(y) {
        *p = p.wrapping_add(*q);
    }
}

#[inline(always)]
fn xor_rotl4(x: &mut [u32; 4], y: &[u32; 4], r: u32) {
    for (p, q) in x.iter_mut().zip(y) {
        *p = (*p ^ *q).rotate_left(r);
    }
}

/// One ChaCha quarter round applied to four independent blocks at once (one per lane), which
/// the compiler turns into vector instructions.
#[inline(always)]
fn qr4(s: &mut [[u32; 4]; 16], a: usize, b: usize, c: usize, d: usize) {
    let (mut va, mut vb, mut vc, mut vd) = (s[a], s[b], s[c], s[d]);
    add4(&mut va, &vb);
    xor_rotl4(&mut vd, &va, 16);
    add4(&mut vc, &vd);
    xor_rotl4(&mut vb, &vc, 12);
    add4(&mut va, &vb);
    xor_rotl4(&mut vd, &va, 8);
    add4(&mut vc, &vd);
    xor_rotl4(&mut vb, &vc, 7);
    s[a] = va;
    s[b] = vb;
    s[c] = vc;
    s[d] = vd;
}

/// Keystream for blocks `counter .. counter + 4`, as 256 bytes.
fn block4(base: &[u32; 16], counter: u32, out: &mut [u8; 256]) {
    let mut init = [[0u32; 4]; 16];
    for i in 0..16 {
        init[i] = [base[i]; 4];
    }
    for (l, v) in init[12].iter_mut().enumerate() {
        *v = counter.wrapping_add(l as u32);
    }
    let mut s = init;
    for _ in 0..10 {
        qr4(&mut s, 0, 4, 8, 12);
        qr4(&mut s, 1, 5, 9, 13);
        qr4(&mut s, 2, 6, 10, 14);
        qr4(&mut s, 3, 7, 11, 15);
        qr4(&mut s, 0, 5, 10, 15);
        qr4(&mut s, 1, 6, 11, 12);
        qr4(&mut s, 2, 7, 8, 13);
        qr4(&mut s, 3, 4, 9, 14);
    }
    for l in 0..4 {
        for i in 0..16 {
            let w = s[i][l].wrapping_add(init[i][l]);
            out[64 * l + 4 * i..64 * l + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
    }
    for row in s.iter_mut().chain(init.iter_mut()) {
        wipe_words(row);
    }
}

/// XOR `data` with the keystream starting at block `counter`.
fn chacha_xor(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN], counter: u32, data: &mut [u8]) {
    assert!(
        (data.len() / 64) + 4 < (u32::MAX - counter) as usize,
        "message too long for a 32-bit block counter"
    );
    let mut base = init_state(key, nonce);
    let mut ks = [0u8; 256];
    let mut ctr = counter;
    for chunk in data.chunks_mut(256) {
        block4(&base, ctr, &mut ks);
        for (d, k) in chunk.iter_mut().zip(ks.iter()) {
            *d ^= *k;
        }
        ctr = ctr.wrapping_add(4);
    }
    // keystream XOR ciphertext is plaintext, and the state holds the key
    sys::wipe(&mut ks);
    wipe_words(&mut base);
}

// ---------------------------------------------------------------------------
// Poly1305, 26-bit limbs
// ---------------------------------------------------------------------------

struct Poly1305 {
    r: [u32; 5],
    h: [u32; 5],
    pad: [u32; 4],
    buf: [u8; 16],
    buf_len: usize,
}

impl Poly1305 {
    fn new(key: &[u8; 32]) -> Self {
        let t0 = le32(key, 0);
        let t1 = le32(key, 4);
        let t2 = le32(key, 8);
        let t3 = le32(key, 12);
        Self {
            r: [
                t0 & 0x03ff_ffff,
                ((t0 >> 26) | (t1 << 6)) & 0x03ff_ff03,
                ((t1 >> 20) | (t2 << 12)) & 0x03ff_c0ff,
                ((t2 >> 14) | (t3 << 18)) & 0x03f0_3fff,
                (t3 >> 8) & 0x000f_ffff,
            ],
            h: [0; 5],
            pad: [le32(key, 16), le32(key, 20), le32(key, 24), le32(key, 28)],
            buf: [0; 16],
            buf_len: 0,
        }
    }

    fn block(&mut self, m: &[u8; 16], hibit: u32) {
        const MASK: u32 = 0x03ff_ffff;
        let t0 = le32(m, 0);
        let t1 = le32(m, 4);
        let t2 = le32(m, 8);
        let t3 = le32(m, 12);

        let h0 = self.h[0].wrapping_add(t0 & MASK);
        let h1 = self.h[1].wrapping_add(((t0 >> 26) | (t1 << 6)) & MASK);
        let h2 = self.h[2].wrapping_add(((t1 >> 20) | (t2 << 12)) & MASK);
        let h3 = self.h[3].wrapping_add(((t2 >> 14) | (t3 << 18)) & MASK);
        let h4 = self.h[4].wrapping_add((t3 >> 8) | hibit);

        let [r0, r1, r2, r3, r4] = self.r.map(u64::from);
        let (s1, s2, s3, s4) = (r1 * 5, r2 * 5, r3 * 5, r4 * 5);
        let [h0, h1, h2, h3, h4] = [h0, h1, h2, h3, h4].map(u64::from);

        let mut d0 = h0 * r0 + h1 * s4 + h2 * s3 + h3 * s2 + h4 * s1;
        let mut d1 = h0 * r1 + h1 * r0 + h2 * s4 + h3 * s3 + h4 * s2;
        let mut d2 = h0 * r2 + h1 * r1 + h2 * r0 + h3 * s4 + h4 * s3;
        let mut d3 = h0 * r3 + h1 * r2 + h2 * r1 + h3 * r0 + h4 * s4;
        let mut d4 = h0 * r4 + h1 * r3 + h2 * r2 + h3 * r1 + h4 * r0;

        let mut c = d0 >> 26;
        let mut n0 = (d0 & MASK as u64) as u32;
        d1 += c;
        c = d1 >> 26;
        let n1 = (d1 & MASK as u64) as u32;
        d2 += c;
        c = d2 >> 26;
        let n2 = (d2 & MASK as u64) as u32;
        d3 += c;
        c = d3 >> 26;
        let n3 = (d3 & MASK as u64) as u32;
        d4 += c;
        c = d4 >> 26;
        let n4 = (d4 & MASK as u64) as u32;
        d0 = n0 as u64 + c * 5;
        c = d0 >> 26;
        n0 = (d0 & MASK as u64) as u32;
        self.h = [n0, n1.wrapping_add(c as u32), n2, n3, n4];
    }

    fn update(&mut self, mut data: &[u8]) {
        if self.buf_len > 0 {
            let take = (16 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len < 16 {
                return;
            }
            let b = self.buf;
            self.block(&b, 1 << 24);
            self.buf_len = 0;
        }
        let mut chunks = data.chunks_exact(16);
        for c in &mut chunks {
            let b: [u8; 16] = c.try_into().unwrap();
            self.block(&b, 1 << 24);
        }
        let rest = chunks.remainder();
        self.buf[..rest.len()].copy_from_slice(rest);
        self.buf_len = rest.len();
    }

    fn finalize(mut self) -> [u8; 16] {
        const MASK: u32 = 0x03ff_ffff;
        if self.buf_len > 0 {
            let mut b = [0u8; 16];
            b[..self.buf_len].copy_from_slice(&self.buf[..self.buf_len]);
            b[self.buf_len] = 1;
            self.block(&b, 0);
        }
        let [mut h0, mut h1, mut h2, mut h3, mut h4] = self.h;

        let mut c = h1 >> 26;
        h1 &= MASK;
        h2 = h2.wrapping_add(c);
        c = h2 >> 26;
        h2 &= MASK;
        h3 = h3.wrapping_add(c);
        c = h3 >> 26;
        h3 &= MASK;
        h4 = h4.wrapping_add(c);
        c = h4 >> 26;
        h4 &= MASK;
        h0 = h0.wrapping_add(c * 5);
        c = h0 >> 26;
        h0 &= MASK;
        h1 = h1.wrapping_add(c);

        let mut g0 = h0.wrapping_add(5);
        c = g0 >> 26;
        g0 &= MASK;
        let mut g1 = h1.wrapping_add(c);
        c = g1 >> 26;
        g1 &= MASK;
        let mut g2 = h2.wrapping_add(c);
        c = g2 >> 26;
        g2 &= MASK;
        let mut g3 = h3.wrapping_add(c);
        c = g3 >> 26;
        g3 &= MASK;
        let g4 = h4.wrapping_add(c).wrapping_sub(1 << 26);

        let mut mask = (g4 >> 31).wrapping_sub(1);
        g0 &= mask;
        g1 &= mask;
        g2 &= mask;
        g3 &= mask;
        let g4 = g4 & mask;
        mask = !mask;
        h0 = (h0 & mask) | g0;
        h1 = (h1 & mask) | g1;
        h2 = (h2 & mask) | g2;
        h3 = (h3 & mask) | g3;
        h4 = (h4 & mask) | g4;

        let w0 = (h0 | (h1 << 26)) as u64;
        let w1 = ((h1 >> 6) | (h2 << 20)) as u64;
        let w2 = ((h2 >> 12) | (h3 << 14)) as u64;
        let w3 = ((h3 >> 18) | (h4 << 8)) as u64;

        let mut f = w0 + self.pad[0] as u64;
        let t0 = f as u32;
        f = w1 + self.pad[1] as u64 + (f >> 32);
        let t1 = f as u32;
        f = w2 + self.pad[2] as u64 + (f >> 32);
        let t2 = f as u32;
        f = w3 + self.pad[3] as u64 + (f >> 32);
        let t3 = f as u32;

        let mut tag = [0u8; 16];
        tag[0..4].copy_from_slice(&t0.to_le_bytes());
        tag[4..8].copy_from_slice(&t1.to_le_bytes());
        tag[8..12].copy_from_slice(&t2.to_le_bytes());
        tag[12..16].copy_from_slice(&t3.to_le_bytes());
        wipe_words(&mut self.r);
        wipe_words(&mut self.h);
        wipe_words(&mut self.pad);
        tag
    }
}

// ---------------------------------------------------------------------------
// AEAD (RFC 8439 section 2.8), in place, with the tag kept separately
// ---------------------------------------------------------------------------

fn poly_key(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN]) -> [u8; 32] {
    let mut base = init_state(key, nonce);
    let mut ks = [0u8; 256];
    block4(&base, 0, &mut ks);
    let mut k = [0u8; 32];
    k.copy_from_slice(&ks[..32]);
    sys::wipe(&mut ks);
    wipe_words(&mut base);
    k
}

fn compute_tag(pk: &[u8; 32], aad: &[&[u8]], ciphertext: &[u8]) -> [u8; TAG_LEN] {
    fn pad16(p: &mut Poly1305, len: usize) {
        let pad = (16 - len % 16) % 16;
        p.update(&[0u8; 16][..pad]);
    }
    let mut p = Poly1305::new(pk);
    let mut aad_len = 0usize;
    for part in aad {
        p.update(part);
        aad_len += part.len();
    }
    pad16(&mut p, aad_len);
    p.update(ciphertext);
    pad16(&mut p, ciphertext.len());
    p.update(&(aad_len as u64).to_le_bytes());
    p.update(&(ciphertext.len() as u64).to_le_bytes());
    p.finalize()
}

fn tags_equal(a: &[u8; TAG_LEN], b: &[u8; TAG_LEN]) -> bool {
    let mut diff = 0u8;
    for i in 0..TAG_LEN {
        diff |= a[i] ^ b[i];
    }
    std::hint::black_box(diff) == 0
}

/// Encrypt `data` in place and return its authentication tag.
pub fn seal_in_place(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN], aad: &[&[u8]], data: &mut [u8]) -> [u8; TAG_LEN] {
    let mut pk = poly_key(key, nonce);
    chacha_xor(key, nonce, 1, data);
    let tag = compute_tag(&pk, aad, data);
    sys::wipe(&mut pk);
    tag
}

/// Verify `tag` over the ciphertext in `data` and, only if it matches, decrypt in place.
/// On failure `data` is left untouched and no plaintext is produced.
pub fn open_in_place(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN], aad: &[&[u8]], data: &mut [u8], tag: &[u8; TAG_LEN]) -> bool {
    let mut pk = poly_key(key, nonce);
    let expected = compute_tag(&pk, aad, data);
    sys::wipe(&mut pk);
    if !tags_equal(&expected, tag) {
        return false;
    }
    chacha_xor(key, nonce, 1, data);
    true
}

// ---------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------

/// A 32-byte key that is wiped on drop and never printed.
pub struct SecretKey {
    bytes: [u8; KEY_LEN],
}

impl SecretKey {
    pub fn from_bytes(bytes: [u8; KEY_LEN]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8; KEY_LEN] {
        &self.bytes
    }

    /// Parse 64 hex characters (surrounding whitespace is ignored).
    pub fn from_hex(text: &[u8]) -> Result<Self, String> {
        let t = text.trim_ascii();
        if t.len() != KEY_LEN * 2 {
            return Err(format!("expected {} hex characters, found {}", KEY_LEN * 2, t.len()));
        }
        let nibble = |c: u8, at: usize| -> Result<u8, String> {
            match c {
                b'0'..=b'9' => Ok(c - b'0'),
                b'a'..=b'f' => Ok(c - b'a' + 10),
                b'A'..=b'F' => Ok(c - b'A' + 10),
                _ => Err(format!("non-hex character at position {at}")),
            }
        };
        let mut bytes = [0u8; KEY_LEN];
        for (i, pair) in t.chunks_exact(2).enumerate() {
            bytes[i] = (nibble(pair[0], 2 * i)? << 4) | nibble(pair[1], 2 * i + 1)?;
        }
        Ok(Self { bytes })
    }

    pub fn from_file(path: &str) -> Result<Self, String> {
        let mut raw = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let key = Self::from_hex(&raw);
        sys::wipe(&mut raw);
        key
    }

    /// `--key-file <path>` if given, otherwise the `LAYERSTREAM_KEY` environment variable.
    pub fn from_cli(key_file: Option<&str>) -> Result<Self, String> {
        match key_file {
            Some(p) => Self::from_file(p),
            None => {
                let text = std::env::var("LAYERSTREAM_KEY")
                    .map_err(|_| "no key given: pass --key-file <path> or set LAYERSTREAM_KEY to 64 hex characters".to_string())?;
                let mut raw = text.into_bytes();
                let key = Self::from_hex(&raw);
                sys::wipe(&mut raw);
                key
            }
        }
    }

    pub fn random() -> Result<Self, String> {
        use std::io::Read;
        let mut bytes = [0u8; KEY_LEN];
        File::open("/dev/urandom")
            .and_then(|mut f| f.read_exact(&mut bytes))
            .map_err(|e| format!("/dev/urandom: {e}"))?;
        Ok(Self { bytes })
    }

    /// Write the key as hex to a new file (mode 0600). Refuses to overwrite.
    pub fn write_to_file(&self, path: &str) -> Result<(), String> {
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .map_err(|e| format!("{path}: {e}"))?;
        let mut line = Vec::with_capacity(KEY_LEN * 2 + 1);
        for b in self.bytes {
            line.push(b"0123456789abcdef"[(b >> 4) as usize]);
            line.push(b"0123456789abcdef"[(b & 15) as usize]);
        }
        line.push(b'\n');
        let r = f.write_all(&line).map_err(|e| format!("{path}: {e}"));
        sys::wipe(&mut line);
        r
    }
}

impl Drop for SecretKey {
    fn drop(&mut self) {
        sys::wipe(&mut self.bytes);
    }
}

impl std::fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SecretKey(<redacted>)")
    }
}

// ---------------------------------------------------------------------------
// Layout of a sealed file
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"LSS1";
const VERSION: u32 = 1;
const FRAME_LEN: usize = 20;

#[derive(Debug, Clone, Copy)]
pub struct SealParams {
    /// Bytes per chunk within a layer block.
    pub layer_chunk: usize,
    /// Embedding rows per chunk.
    pub embed_chunk_rows: usize,
}

impl Default for SealParams {
    fn default() -> Self {
        SealParams { layer_chunk: 1 << 20, embed_chunk_rows: 512 }
    }
}

#[derive(Debug, Clone)]
struct SealedLayout {
    config_text: String,
    cfg: Config,
    layout: LayerLayout,
    params: SealParams,
    final_norm_bytes: usize,
}

impl SealedLayout {
    fn layer_bytes(&self) -> usize {
        self.layout.total
    }

    fn embed_bytes(&self) -> usize {
        self.cfg.vocab * self.cfg.row_bytes()
    }

    fn layer_chunks(&self) -> usize {
        self.layer_bytes().div_ceil(self.params.layer_chunk)
    }

    fn embed_chunks(&self) -> usize {
        self.cfg.vocab.div_ceil(self.params.embed_chunk_rows)
    }

    fn embed_first_chunk(&self) -> usize {
        self.cfg.layers * self.layer_chunks()
    }

    fn final_chunk(&self) -> usize {
        self.embed_first_chunk() + self.embed_chunks()
    }

    fn n_chunks(&self) -> usize {
        self.final_chunk() + 1
    }

    fn embed_plain_off(&self) -> u64 {
        (self.cfg.layers * self.layer_bytes()) as u64
    }

    fn total_plain(&self) -> u64 {
        self.embed_plain_off() + self.embed_bytes() as u64 + self.final_norm_bytes as u64
    }

    fn to_json(&self) -> String {
        let mut ranges = String::new();
        for (i, (o, l)) in self.layout.ranges.iter().enumerate() {
            if i > 0 {
                ranges.push(',');
            }
            let _ = write!(ranges, "[{o},{l}]");
        }
        format!(
            "{{\"format\":\"layerstream-sealed-1\",\"config_json\":{},\"layer_bytes\":{},\"layer_chunk\":{},\"embed_chunk_rows\":{},\"final_norm_bytes\":{},\"layer_layout\":[{}]}}",
            json_string(&self.config_text),
            self.layer_bytes(),
            self.params.layer_chunk,
            self.params.embed_chunk_rows,
            self.final_norm_bytes,
            ranges
        )
    }

    fn from_json(text: &str) -> Result<SealedLayout, String> {
        let j = Json::parse(text)?;
        let u = |k: &str| -> Result<usize, String> {
            j.get(k).and_then(Json::as_u64).map(|v| v as usize).ok_or_else(|| format!("sealed header: missing {k}"))
        };
        if j.get("format").and_then(Json::as_str) != Some("layerstream-sealed-1") {
            return Err("unknown sealed format".into());
        }
        let config_text = j.get("config_json").and_then(Json::as_str).ok_or("sealed header: missing config")?.to_string();
        let cfg = Config::from_json(&Json::parse(&config_text)?)?;
        let ranges_json = j.get("layer_layout").and_then(Json::as_arr).ok_or("sealed header: missing layout")?;
        if ranges_json.len() != 12 {
            return Err("sealed header: layout must list 12 tensors".into());
        }
        let mut ranges = [(0usize, 0usize); 12];
        let mut total = 0usize;
        for (i, r) in ranges_json.iter().enumerate() {
            let pair = r.as_arr().filter(|a| a.len() == 2).ok_or("sealed header: bad layout entry")?;
            let (o, l) = (pair[0].as_u64().ok_or("bad offset")? as usize, pair[1].as_u64().ok_or("bad length")? as usize);
            total = total.max(o + l);
            ranges[i] = (o, l);
        }
        let layer_bytes = u("layer_bytes")?;
        if layer_bytes != total {
            return Err("sealed header: layer layout does not add up".into());
        }
        let params = SealParams { layer_chunk: u("layer_chunk")?, embed_chunk_rows: u("embed_chunk_rows")? };
        if params.layer_chunk == 0 || params.embed_chunk_rows == 0 {
            return Err("sealed header: zero chunk size".into());
        }
        let final_norm_bytes = u("final_norm_bytes")?;
        if final_norm_bytes != cfg.hidden * 2 || cfg.layers == 0 || cfg.layers > 4096 {
            return Err("sealed header: inconsistent sizes".into());
        }
        Ok(SealedLayout { config_text, cfg, layout: LayerLayout { ranges, total }, params, final_norm_bytes })
    }

    fn chunk_len(&self, idx: usize) -> usize {
        let embed_first = self.embed_first_chunk();
        if idx < embed_first {
            let k = idx % self.layer_chunks();
            self.params.layer_chunk.min(self.layer_bytes() - k * self.params.layer_chunk)
        } else if idx < self.final_chunk() {
            let rb = self.cfg.row_bytes();
            let k = idx - embed_first;
            let chunk = self.params.embed_chunk_rows * rb;
            chunk.min(self.embed_bytes() - k * chunk)
        } else {
            self.final_norm_bytes
        }
    }
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn chunk_nonce(prefix: &[u8; 8], idx: usize) -> [u8; NONCE_LEN] {
    let mut n = [0u8; NONCE_LEN];
    n[..8].copy_from_slice(prefix);
    n[8..].copy_from_slice(&(idx as u32).to_le_bytes());
    n
}

/// Run `f(k, chunk_k, extra_k)` for every `chunk_size` chunk of `data` across `threads`
/// threads. Returns `Err(k)` for the lowest chunk index whose call returned `false`.
fn for_each_chunk<T: Send>(
    data: &mut [u8],
    chunk_size: usize,
    extra: &mut [T],
    threads: usize,
    f: &(dyn Fn(usize, &mut [u8], &mut T) -> bool + Sync),
) -> Result<(), usize> {
    let n = data.len().div_ceil(chunk_size);
    assert_eq!(extra.len(), n, "one extra slot per chunk");
    let items = Mutex::new(data.chunks_mut(chunk_size).zip(extra.iter_mut()).enumerate());
    let failed: Mutex<Option<usize>> = Mutex::new(None);
    let work = || loop {
        let item = items.lock().unwrap_or_else(|p| p.into_inner()).next();
        let Some((k, (chunk, ex))) = item else { break };
        if !f(k, chunk, ex) {
            let mut fl = failed.lock().unwrap_or_else(|p| p.into_inner());
            *fl = Some(fl.map_or(k, |p| p.min(k)));
            break;
        }
    };
    if threads <= 1 || n <= 1 {
        work();
    } else {
        std::thread::scope(|s| {
            for _ in 0..threads.min(n) {
                s.spawn(work);
            }
        });
    }
    match failed.into_inner().unwrap_or_else(|p| p.into_inner()) {
        Some(k) => Err(k),
        None => Ok(()),
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

// ---------------------------------------------------------------------------
// Sealing a model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct SealSummary {
    pub plain_bytes: u64,
    pub sealed_bytes: u64,
    pub chunks: usize,
}

/// Encrypt the model behind `src` into a new file at `out` (it must not exist).
pub fn seal_model(
    src: &FileSource,
    config_text: &str,
    key: &SecretKey,
    nonce_prefix: [u8; 8],
    params: SealParams,
    out: &Path,
) -> Result<SealSummary, String> {
    let files = src.files();
    let layout = SealedLayout {
        config_text: config_text.to_string(),
        cfg: files.cfg.clone(),
        layout: files.layout.clone(),
        params,
        final_norm_bytes: files.norm_len,
    };
    if params.layer_chunk == 0 || params.embed_chunk_rows == 0 {
        return Err("chunk sizes must be positive".into());
    }
    let header = layout.to_json();
    let header_bytes = header.as_bytes();
    let mut frame = Vec::with_capacity(FRAME_LEN + header_bytes.len());
    frame.extend_from_slice(MAGIC);
    frame.extend_from_slice(&VERSION.to_le_bytes());
    frame.extend_from_slice(&nonce_prefix);
    frame.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    frame.extend_from_slice(header_bytes);

    let n_chunks = layout.n_chunks();
    let tags_off = frame.len() as u64;
    let ct_base = tags_off + (TAG_LEN * n_chunks) as u64;

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out)
        .map_err(|e| format!("{}: {e}", out.display()))?;
    file.write_all(&frame).map_err(|e| e.to_string())?;
    file.write_all(&vec![0u8; TAG_LEN * n_chunks]).map_err(|e| e.to_string())?;

    let threads = default_threads();
    let mut tags: Vec<[u8; TAG_LEN]> = vec![[0u8; TAG_LEN]; n_chunks];
    let encrypt = |first_idx: usize, data: &mut [u8], chunk_size: usize, tag_slots: &mut [[u8; TAG_LEN]]| -> Result<(), String> {
        let f = |k: usize, chunk: &mut [u8], slot: &mut [u8; TAG_LEN]| {
            let idx = first_idx + k;
            let idx_bytes = (idx as u32).to_le_bytes();
            *slot = seal_in_place(key.as_bytes(), &chunk_nonce(&nonce_prefix, idx), &[&frame, &idx_bytes], chunk);
            true
        };
        for_each_chunk(data, chunk_size, tag_slots, threads, &f).map_err(|k| format!("chunk {k} failed"))
    };

    let mut written: u64 = 0;
    let mut layer_buf = vec![0u8; layout.layer_bytes()];
    let lc = layout.layer_chunks();
    for l in 0..layout.cfg.layers {
        src.read_layer(l, &mut layer_buf)?;
        encrypt(l * lc, &mut layer_buf, params.layer_chunk, &mut tags[l * lc..(l + 1) * lc])?;
        file.write_all(&layer_buf).map_err(|e| e.to_string())?;
        written += layer_buf.len() as u64;
    }
    sys::wipe(&mut layer_buf);
    drop(layer_buf);

    let rb = layout.cfg.row_bytes();
    let group_rows = params.embed_chunk_rows * 16;
    let mut emb_buf = vec![0u8; group_rows * rb];
    let mut first = 0;
    while first < layout.cfg.vocab {
        let rows = group_rows.min(layout.cfg.vocab - first);
        let buf = &mut emb_buf[..rows * rb];
        src.read_embed_rows(first, rows, buf)?;
        let first_chunk = layout.embed_first_chunk() + first / params.embed_chunk_rows;
        let n = rows.div_ceil(params.embed_chunk_rows);
        encrypt(first_chunk, buf, params.embed_chunk_rows * rb, &mut tags[first_chunk..first_chunk + n])?;
        file.write_all(buf).map_err(|e| e.to_string())?;
        written += buf.len() as u64;
        first += rows;
    }
    sys::wipe(&mut emb_buf);

    let mut norm = vec![0u8; layout.final_norm_bytes];
    src.read_final_norm_raw(&mut norm)?;
    let fc = layout.final_chunk();
    let norm_len = norm.len();
    encrypt(fc, &mut norm, norm_len, &mut tags[fc..fc + 1])?;
    file.write_all(&norm).map_err(|e| e.to_string())?;
    written += norm.len() as u64;

    if written != layout.total_plain() {
        return Err(format!("wrote {written} bytes but expected {}", layout.total_plain()));
    }
    let tag_bytes: Vec<u8> = tags.iter().flatten().copied().collect();
    file.write_all_at(&tag_bytes, tags_off).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())?;
    Ok(SealSummary { plain_bytes: written, sealed_bytes: ct_base + written, chunks: n_chunks })
}

/// A random 8-byte nonce prefix from the operating system.
pub fn random_nonce_prefix() -> Result<[u8; 8], String> {
    use std::io::Read;
    let mut p = [0u8; 8];
    File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut p)).map_err(|e| format!("/dev/urandom: {e}"))?;
    Ok(p)
}

// ---------------------------------------------------------------------------
// Reading a sealed model
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct OpenOptions {
    pub throttle: Option<Throttle>,
    /// Threads used to decrypt chunks of one block in parallel (default: all cores).
    pub threads: Option<usize>,
}

pub struct SealedSource {
    file: File,
    key: SecretKey,
    layout: SealedLayout,
    nonce_prefix: [u8; 8],
    aad_prefix: Vec<u8>,
    tags: Vec<[u8; TAG_LEN]>,
    ct_base: u64,
    threads: usize,
    throttle: Option<Throttle>,
    final_norm: Vec<f32>,
    bytes: AtomicU64,
    read_ns: AtomicU64,
    decrypt_ns: AtomicU64,
}

impl SealedSource {
    pub fn open(path: &Path, key: SecretKey, opts: OpenOptions) -> Result<SealedSource, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file_len = file.metadata().map_err(|e| e.to_string())?.len();
        let mut frame = [0u8; FRAME_LEN];
        file.read_exact_at(&mut frame, 0).map_err(|_| "file too short to be a sealed model".to_string())?;
        if &frame[..4] != MAGIC {
            return Err("not a sealed layerstream model (bad magic bytes)".into());
        }
        let version = u32::from_le_bytes(frame[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(format!("unsupported sealed format version {version}"));
        }
        let nonce_prefix: [u8; 8] = frame[8..16].try_into().unwrap();
        let header_len = u32::from_le_bytes(frame[16..20].try_into().unwrap()) as usize;
        if header_len == 0 || header_len > (1 << 20) || (FRAME_LEN + header_len) as u64 > file_len {
            return Err("implausible sealed header length".into());
        }
        let mut aad_prefix = vec![0u8; FRAME_LEN + header_len];
        file.read_exact_at(&mut aad_prefix, 0).map_err(|e| e.to_string())?;
        let header_text = std::str::from_utf8(&aad_prefix[FRAME_LEN..]).map_err(|_| "header is not UTF-8".to_string())?;
        let layout = SealedLayout::from_json(header_text)?;

        let n_chunks = layout.n_chunks();
        let tags_off = aad_prefix.len() as u64;
        let ct_base = tags_off + (TAG_LEN * n_chunks) as u64;
        if file_len != ct_base + layout.total_plain() {
            return Err(format!("file is {file_len} bytes but its header implies {}", ct_base + layout.total_plain()));
        }
        let mut raw_tags = vec![0u8; TAG_LEN * n_chunks];
        file.read_exact_at(&mut raw_tags, tags_off).map_err(|e| e.to_string())?;
        let tags: Vec<[u8; TAG_LEN]> = raw_tags.chunks_exact(TAG_LEN).map(|c| c.try_into().unwrap()).collect();

        let mut src = SealedSource {
            file,
            key,
            layout,
            nonce_prefix,
            aad_prefix,
            tags,
            ct_base,
            threads: opts.threads.unwrap_or_else(default_threads).max(1),
            throttle: opts.throttle,
            final_norm: Vec::new(),
            bytes: AtomicU64::new(0),
            read_ns: AtomicU64::new(0),
            decrypt_ns: AtomicU64::new(0),
        };

        // Decrypting the final norm now checks the key (and part of the file) up front.
        let mut raw = vec![0u8; src.layout.final_norm_bytes];
        let fc = src.layout.final_chunk();
        let off = src.layout.total_plain() - src.layout.final_norm_bytes as u64;
        src.read_ciphertext(&mut raw, off)?;
        if !src.open_chunk(fc, &mut raw) {
            return Err("authentication failed: wrong key, or the file was modified".into());
        }
        let mut fnorm = vec![0.0f32; src.layout.cfg.hidden];
        crate::kernels::bf16_slice_to_f32(&raw, &mut fnorm);
        sys::wipe(&mut raw);
        src.final_norm = fnorm;
        Ok(src)
    }

    fn read_ciphertext(&self, buf: &mut [u8], plain_off: u64) -> Result<(), String> {
        let deadline = self.throttle.as_ref().map(|t| t.reserve(buf.len()));
        let t = Instant::now();
        self.file.read_exact_at(buf, self.ct_base + plain_off).map_err(|e| format!("read: {e}"))?;
        if let Some(d) = deadline {
            Throttle::wait_until(d);
        }
        self.bytes.fetch_add(buf.len() as u64, Ordering::Relaxed);
        self.read_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn open_chunk(&self, idx: usize, data: &mut [u8]) -> bool {
        let idx_bytes = (idx as u32).to_le_bytes();
        open_in_place(
            self.key.as_bytes(),
            &chunk_nonce(&self.nonce_prefix, idx),
            &[&self.aad_prefix, &idx_bytes],
            data,
            &self.tags[idx],
        )
    }

    /// Decrypt consecutive chunks in place; `data` must start on a chunk boundary.
    fn decrypt_range(&self, first_chunk: usize, data: &mut [u8], chunk_size: usize) -> Result<(), String> {
        let t = Instant::now();
        let n = data.len().div_ceil(chunk_size);
        let mut slots = vec![(); n];
        let f = |k: usize, chunk: &mut [u8], _: &mut ()| self.open_chunk(first_chunk + k, chunk);
        let r = for_each_chunk(data, chunk_size, &mut slots, self.threads, &f);
        self.decrypt_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r.map_err(|k| format!("authentication failed for chunk {}: wrong key, or the file was modified", first_chunk + k))
    }
}

impl Source for SealedSource {
    fn cfg(&self) -> &Config {
        &self.layout.cfg
    }

    fn layout(&self) -> &LayerLayout {
        &self.layout.layout
    }

    fn final_norm(&self) -> &[f32] {
        &self.final_norm
    }

    fn read_layer(&self, l: usize, buf: &mut [u8]) -> Result<(), String> {
        if l >= self.layout.cfg.layers {
            return Err(format!("layer {l} out of range"));
        }
        if buf.len() != self.layout.layer_bytes() {
            return Err("layer buffer has the wrong size".into());
        }
        self.read_ciphertext(buf, (l * self.layout.layer_bytes()) as u64)?;
        self.decrypt_range(l * self.layout.layer_chunks(), buf, self.layout.params.layer_chunk)
    }

    fn read_embed_rows(&self, first: usize, rows: usize, buf: &mut [u8]) -> Result<(), String> {
        let cfg = &self.layout.cfg;
        let rb = cfg.row_bytes();
        if first + rows > cfg.vocab || buf.len() != rows * rb {
            return Err("embedding range out of bounds".into());
        }
        if rows == 0 {
            return Ok(());
        }
        let ecr = self.layout.params.embed_chunk_rows;
        let embed_first = self.layout.embed_first_chunk();
        let aligned = first.is_multiple_of(ecr) && (rows.is_multiple_of(ecr) || first + rows == cfg.vocab);
        if aligned {
            self.read_ciphertext(buf, self.layout.embed_plain_off() + (first * rb) as u64)?;
            return self.decrypt_range(embed_first + first / ecr, buf, ecr * rb);
        }
        // Unaligned: decrypt each overlapping chunk on its own and copy out the rows we need.
        let (c0, c1) = (first / ecr, (first + rows - 1) / ecr);
        for c in c0..=c1 {
            let idx = embed_first + c;
            let mut tmp = vec![0u8; self.layout.chunk_len(idx)];
            self.read_ciphertext(&mut tmp, self.layout.embed_plain_off() + (c * ecr * rb) as u64)?;
            self.decrypt_range(idx, &mut tmp, ecr * rb)?;
            let (row0, row1) = ((c * ecr).max(first), ((c + 1) * ecr).min(first + rows));
            let src = &tmp[(row0 - c * ecr) * rb..(row1 - c * ecr) * rb];
            buf[(row0 - first) * rb..(row1 - first) * rb].copy_from_slice(src);
            sys::wipe(&mut tmp);
        }
        Ok(())
    }

    fn stats(&self) -> IoStats {
        IoStats {
            bytes_read: self.bytes.load(Ordering::Relaxed),
            read_ns: self.read_ns.load(Ordering::Relaxed),
            decrypt_ns: self.decrypt_ns.load(Ordering::Relaxed),
        }
    }

    fn describe(&self) -> String {
        let mut s = format!("sealed, ChaCha20-Poly1305, {} decrypt threads", self.threads);
        if let Some(t) = &self.throttle {
            s.push_str(&format!(", storage capped at {:.0} MB/s", t.mb_per_s()));
        }
        s
    }
}

/// Check that two sources hold exactly the same model (every layer, the whole embedding
/// table, the final norm).
pub fn verify_same(a: &dyn Source, b: &dyn Source) -> Result<(), String> {
    if a.cfg() != b.cfg() || a.layout() != b.layout() {
        return Err("sources disagree about the model's shape".into());
    }
    let cfg = a.cfg().clone();
    let (mut x, mut y) = (vec![0u8; a.layout().total], vec![0u8; a.layout().total]);
    for l in 0..cfg.layers {
        a.read_layer(l, &mut x)?;
        b.read_layer(l, &mut y)?;
        if x != y {
            return Err(format!("layer {l} differs"));
        }
    }
    let rows = 4096.min(cfg.vocab);
    let rb = cfg.row_bytes();
    let (mut x, mut y) = (vec![0u8; rows * rb], vec![0u8; rows * rb]);
    let mut first = 0;
    while first < cfg.vocab {
        let n = rows.min(cfg.vocab - first);
        a.read_embed_rows(first, n, &mut x[..n * rb])?;
        b.read_embed_rows(first, n, &mut y[..n * rb])?;
        if x[..n * rb] != y[..n * rb] {
            return Err(format!("embedding rows {first}..{} differ", first + n));
        }
        first += n;
    }
    if a.final_norm() != b.final_norm() {
        return Err("final norm differs".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
    }

    // Vectors come from RFC 8439 and were confirmed with Python's `cryptography` package.

    #[test]
    fn chacha20_block_matches_rfc8439() {
        let key: [u8; 32] = std::array::from_fn(|i| i as u8);
        let nonce: [u8; 12] = unhex("000000090000004a00000000").try_into().unwrap();
        let base = init_state(&key, &nonce);
        let mut ks = [0u8; 256];
        block4(&base, 1, &mut ks);
        assert_eq!(
            ks[..64].to_vec(),
            unhex(
                "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e\
                 d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e"
            )
        );
        // The other three lanes are the following blocks: counters 2, 3 and 4.
        let mut again = [0u8; 256];
        block4(&base, 2, &mut again);
        assert_eq!(ks[64..], again[..192]);
    }

    #[test]
    fn poly1305_matches_rfc8439() {
        let key: [u8; 32] = unhex("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b").try_into().unwrap();
        let mut p = Poly1305::new(&key);
        p.update(b"Cryptographic Forum Research Group");
        assert_eq!(p.finalize().to_vec(), unhex("a8061dc1305136c6c22b8baf0c0127a9"));
    }

    #[test]
    fn aead_matches_rfc8439_section_2_8_2() {
        let key: [u8; 32] = std::array::from_fn(|i| 0x80 + i as u8);
        let nonce: [u8; 12] = unhex("070000004041424344454647").try_into().unwrap();
        let aad = unhex("50515253c0c1c2c3c4c5c6c7");
        let pt = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
        let mut data = pt.to_vec();
        let tag = seal_in_place(&key, &nonce, &[&aad], &mut data);
        assert_eq!(&data[..16], &unhex("d31a8d34648e60db7b86afbc53ef7ec2")[..]);
        assert_eq!(tag.to_vec(), unhex("1ae10b594f09e26a7e902ecbd0600691"));
        assert!(open_in_place(&key, &nonce, &[&aad], &mut data, &tag));
        assert_eq!(&data[..], &pt[..]);
    }

    #[test]
    fn matches_python_for_many_lengths() {
        let key: [u8; 32] = std::array::from_fn(|i| (i * 7 + 3) as u8);
        let nonce: [u8; 12] = std::array::from_fn(|i| (i * 11 + 5) as u8);
        let expected = [
            (0, "8dda000cf4057e4b5850f2226b360ac1"),
            (1, "da6e0d0cda7a0f650b2c612bf1a72cfa"),
            (15, "946d3133f7f73d8fd604b80b8d003b02"),
            (16, "1cd4c6f936ba3e21aa7703c912ef8d25"),
            (17, "57d03f332d780fab1a2bb99427cb1f1b"),
            (63, "50495cb084db6408cb13bac23788b67c"),
            (64, "0f835779fea9aa674dedac301ca54c92"),
            (65, "cf3f6c873572f558c1c91bc21c6618ac"),
            (127, "888fc69f6549d3e8c4dfab7b714937e1"),
            (128, "9ec3e08dc25e92393816bf0ea506b572"),
            (129, "0691ec77949df8155c2ee121325bec51"),
            (255, "7402ca82aa8e459b472c372c0f0f7b7b"),
            (256, "c5393742d22d200298f01bce68fcb263"),
            (257, "d98699e528ad496a0db1a0205a6656b8"),
            (511, "971391cd619dcc21f6d211119d35327c"),
            (512, "040e51e2263daecc39a3130fd2be0c81"),
            (513, "4abadc2c859967c5d24b1b9a56e2a3dc"),
            (1000, "43ade245f7420997994e1fbbdcb9d643"),
            (1023, "3106ce499411d319234b7389e2f0bfcf"),
            (1024, "d45cdee0ac91b1e3ae26ecc0a1ee5912"),
            (1025, "44d50f95328cf7aaf34be1fce8ffecdf"),
            (4096, "6a0b1779c32836a856629f97ce9722d7"),
            (65537, "ef04344e637ab22416c544be2b042d0a"),
        ];
        for (len, tag) in expected {
            let pt: Vec<u8> = (0..len).map(|i| ((i * 31 + 7 + (i >> 8) * 3) & 0xff) as u8).collect();
            let aad: Vec<u8> = (0..len % 13).map(|i| ((i * 5 + 1) & 0xff) as u8).collect();
            let mut data = pt.clone();
            let got = seal_in_place(&key, &nonce, &[&aad], &mut data);
            assert_eq!(got.to_vec(), unhex(tag), "tag mismatch at length {len}");
            assert!(open_in_place(&key, &nonce, &[&aad], &mut data, &got));
            assert_eq!(data, pt, "round trip at length {len}");
        }
    }

    #[test]
    fn any_modification_is_rejected_and_nothing_is_decrypted() {
        let key = [7u8; 32];
        let nonce = [9u8; 12];
        let pt: Vec<u8> = (0..700u32).map(|i| i as u8).collect();
        let mut sealed = pt.clone();
        let tag = seal_in_place(&key, &nonce, &[b"hdr"], &mut sealed);
        for pos in (0..sealed.len()).step_by(37).chain([sealed.len() - 1]) {
            let mut bad = sealed.clone();
            bad[pos] ^= 1;
            let snapshot = bad.clone();
            assert!(!open_in_place(&key, &nonce, &[b"hdr"], &mut bad, &tag), "byte {pos}");
            assert_eq!(bad, snapshot, "data must be untouched on failure");
        }
        let mut d = sealed.clone();
        let mut bad_tag = tag;
        bad_tag[0] ^= 1;
        assert!(!open_in_place(&key, &nonce, &[b"hdr"], &mut d, &bad_tag));
        assert!(!open_in_place(&[8u8; 32], &nonce, &[b"hdr"], &mut d, &tag), "wrong key");
        assert!(!open_in_place(&key, &[1u8; 12], &[b"hdr"], &mut d, &tag), "wrong nonce");
        assert!(!open_in_place(&key, &nonce, &[b"other"], &mut d, &tag), "wrong aad");
    }

    #[test]
    fn key_parsing_and_redaction() {
        let hex = "00112233445566778899aabbccddeeff00112233445566778899AABBCCDDEEFF";
        let key = SecretKey::from_hex(format!("  {hex}\n").as_bytes()).unwrap();
        assert_eq!((key.as_bytes()[0], key.as_bytes()[15], key.as_bytes()[31]), (0x00, 0xff, 0xff));
        assert_eq!(format!("{key:?}"), "SecretKey(<redacted>)");
        assert!(SecretKey::from_hex(b"abcd").is_err());
        let err = SecretKey::from_hex(format!("zz{}", &hex[2..]).as_bytes()).unwrap_err();
        assert!(!err.contains(&hex[2..10]), "error must not echo key material: {err}");
    }

    #[test]
    fn chunk_helper_covers_every_chunk_once_and_reports_the_lowest_failure() {
        let mut data = vec![0u8; 1000];
        let mut extra = vec![0usize; 8];
        let ok = for_each_chunk(&mut data, 128, &mut extra, 4, &|k, chunk, e| {
            chunk.fill(k as u8 + 1);
            *e += 1;
            true
        });
        assert_eq!(ok, Ok(()));
        assert!(extra.iter().all(|&e| e == 1));
        assert_eq!(data[999], 8);
        let mut extra = vec![(); 8];
        let r = for_each_chunk(&mut data, 128, &mut extra, 1, &|k, _, _| k != 3 && k != 5);
        assert_eq!(r, Err(3));
    }

    #[test]
    fn json_strings_round_trip_through_the_parser() {
        for s in ["plain", "quote \" and \\ slash", "line\nbreak\ttab", "unicode é 日本 😀", "\u{1}control"] {
            assert_eq!(Json::parse(&json_string(s)).unwrap().as_str(), Some(s));
        }
    }
}
