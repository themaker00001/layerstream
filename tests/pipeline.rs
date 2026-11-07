//! Whole-pipeline tests on a tiny synthetic model: no download needed.

mod common;

use common::*;
use layerstream::model::{Model, Sampler};
use layerstream::provider::{Provider, ResidentProvider, StreamProvider};
use layerstream::seal::{self, OpenOptions, SealParams, SealedSource, SecretKey};
use layerstream::source::{FileSource, Source, Throttle};
use layerstream::weights::ModelFiles;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn key(n: u8) -> SecretKey {
    SecretKey::from_bytes([n; 32])
}

fn plain(dir: &Path) -> Arc<FileSource> {
    Arc::new(FileSource::open(ModelFiles::open(dir).unwrap()).unwrap())
}

/// Small chunks so layers and the embedding table each span several chunks, and the last
/// chunk of each is partial.
const SMALL: SealParams = SealParams { layer_chunk: 1500, embed_chunk_rows: 64 };

fn seal_into(dir: &Path, k: &SecretKey, params: SealParams) -> std::path::PathBuf {
    let out = dir.join("model.sealed");
    let _ = std::fs::remove_file(&out);
    let src = plain(dir);
    let cfg = std::fs::read_to_string(dir.join("config.json")).unwrap();
    seal::seal_model(&src, &cfg, k, *b"abcdefgh", params, &out).unwrap();
    out
}

fn sealed(path: &Path, k: SecretKey) -> Arc<SealedSource> {
    Arc::new(SealedSource::open(path, k, OpenOptions { threads: Some(3), ..Default::default() }).unwrap())
}

fn model_from(p: Box<dyn Provider>) -> Model {
    Model::new(p, 3, 64)
}

const PROMPT: [u32; 6] = [5, 17, 301, 42, 599, 0];

fn run(m: &mut Model) -> Vec<Vec<f32>> {
    let mut out = vec![m.forward(&PROMPT).unwrap()];
    let mut s = Sampler::greedy();
    for _ in 0..3 {
        let next = s.sample(out.last().unwrap());
        out.push(m.forward(&[next]).unwrap());
    }
    out
}

fn bit_equal(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()))
}

#[test]
fn sealed_file_decrypts_to_exactly_the_original_weights() {
    let dir = temp_dir("seal-roundtrip");
    write_tiny_model(&dir, 1);
    let k = key(7);
    for params in [SMALL, SealParams::default(), SealParams { layer_chunk: 64, embed_chunk_rows: 7 }] {
        let path = seal_into(&dir, &k, params);
        let s = sealed(&path, key(7));
        seal::verify_same(plain(&dir).as_ref(), s.as_ref()).unwrap();
    }
    // No 24-byte piece of the weights appears in the sealed file.
    let path = seal_into(&dir, &k, SMALL);
    let sealed_bytes = std::fs::read(&path).unwrap();
    let plain_bytes = std::fs::read(dir.join("model.safetensors")).unwrap();
    let start = plain_bytes.len() - 5000; // inside the last layer's weights
    assert!(!sealed_bytes.windows(24).any(|w| w == &plain_bytes[start..start + 24]));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn every_mode_produces_bit_identical_logits() {
    let dir = temp_dir("modes");
    write_tiny_model(&dir, 2);
    let path = seal_into(&dir, &key(9), SMALL);

    let reference = run(&mut model_from(Box::new(ResidentProvider::load(plain(&dir)).unwrap())));
    assert!(reference.iter().flatten().all(|v| v.is_finite()));

    for slots in [1usize, 2, 3] {
        for wipe in [false, true] {
            let m = &mut model_from(Box::new(StreamProvider::new(plain(&dir), slots, wipe)));
            assert!(bit_equal(&reference, &run(m)), "plain stream, {slots} slots, wipe={wipe}");
        }
    }
    let m = &mut model_from(Box::new(ResidentProvider::load(sealed(&path, key(9))).unwrap()));
    assert!(bit_equal(&reference, &run(m)), "sealed resident");
    for slots in [1usize, 2] {
        let m = &mut model_from(Box::new(StreamProvider::new(sealed(&path, key(9)), slots, true)));
        assert!(bit_equal(&reference, &run(m)), "sealed stream, {slots} slots");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn wrong_key_and_tampering_are_rejected() {
    let dir = temp_dir("tamper");
    write_tiny_model(&dir, 3);
    let path = seal_into(&dir, &key(1), SMALL);

    // Wrong key: refused at open.
    let err = SealedSource::open(&path, key(2), OpenOptions::default()).err().expect("wrong key must fail");
    assert!(err.contains("authentication failed"), "{err}");

    let good = std::fs::read(&path).unwrap();
    let total = good.len();
    let flip = |pos: usize, name: &str| {
        let mut bad = good.clone();
        bad[pos] ^= 1;
        let p = dir.join(name);
        std::fs::write(&p, bad).unwrap();
        p
    };

    // A flipped bit inside a layer: the file opens (only the final norm is checked up front)
    // but loading everything, or streaming through that layer, fails.
    let mid = flip(total / 3, "bad-ct.sealed");
    let s = Arc::new(SealedSource::open(&mid, key(1), OpenOptions::default()).unwrap());
    assert!(ResidentProvider::load(s.clone()).is_err(), "resident load must detect it");
    let mut m = model_from(Box::new(StreamProvider::new(s, 2, false)));
    let e = m.forward(&PROMPT).unwrap_err();
    assert!(e.contains("authentication failed"), "{e}");

    // A flipped bit in the header or in the tag table: refused (header) or detected (tags).
    assert!(SealedSource::open(&flip(30, "bad-header.sealed"), key(1), OpenOptions::default()).is_err());
    let tag_pos = 20 + u32::from_le_bytes(good[16..20].try_into().unwrap()) as usize + 5;
    let s = Arc::new(SealedSource::open(&flip(tag_pos, "bad-tag.sealed"), key(1), OpenOptions::default()).unwrap());
    assert!(ResidentProvider::load(s).is_err(), "a changed tag must be detected");

    // Truncated, wrong magic.
    std::fs::write(dir.join("short.sealed"), &good[..total - 1]).unwrap();
    assert!(SealedSource::open(&dir.join("short.sealed"), key(1), OpenOptions::default()).is_err());
    let mut magic = good.clone();
    magic[0] = b'X';
    std::fs::write(dir.join("magic.sealed"), magic).unwrap();
    assert!(SealedSource::open(&dir.join("magic.sealed"), key(1), OpenOptions::default()).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn chunks_cannot_be_swapped() {
    // Two layers have identical sizes, so swapping their ciphertext bytes yields a valid-looking
    // file. The per-chunk nonce and associated data must make it fail to authenticate.
    let dir = temp_dir("swap");
    write_tiny_model(&dir, 4);
    let path = seal_into(&dir, &key(5), SealParams { layer_chunk: 1 << 20, embed_chunk_rows: 64 });
    let good = std::fs::read(&path).unwrap();
    let s = SealedSource::open(&path, key(5), OpenOptions::default()).unwrap();
    let layer = s.layout().total;
    let n_chunks = LAYERS + VOCAB.div_ceil(64) + 1;
    let ct_base = good.len() - (LAYERS * layer + VOCAB * HIDDEN * 2 + HIDDEN * 2);
    assert_eq!(ct_base, 20 + u32::from_le_bytes(good[16..20].try_into().unwrap()) as usize + 16 * n_chunks);
    let mut bad = good.clone();
    bad[ct_base..ct_base + layer].copy_from_slice(&good[ct_base + layer..ct_base + 2 * layer]);
    bad[ct_base + layer..ct_base + 2 * layer].copy_from_slice(&good[ct_base..ct_base + layer]);
    std::fs::write(dir.join("swapped.sealed"), bad).unwrap();
    let s = Arc::new(SealedSource::open(&dir.join("swapped.sealed"), key(5), OpenOptions::default()).unwrap());
    assert!(ResidentProvider::load(s).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn read_ahead_follows_the_schedule_and_rejects_out_of_order_requests() {
    let dir = temp_dir("schedule");
    write_tiny_model(&dir, 5);
    let mut p = StreamProvider::new(plain(&dir), 2, false);
    p.begin_forward(true).unwrap();
    assert!(p.layer(1).is_err(), "layer 1 before layer 0");
    // The provider stays usable: start over and consume everything in order.
    p.begin_forward(true).unwrap();
    for l in 0..LAYERS {
        assert_eq!(p.layer(l).unwrap().len(), p.layout().total);
    }
    let mut rows = 0;
    for c in 0..p.n_head_chunks() {
        rows += p.head_chunk(c).unwrap().rows;
    }
    assert_eq!(rows, VOCAB);
    p.end_forward().unwrap();
    // Ending early is an error, and the next pass recovers.
    p.begin_forward(false).unwrap();
    p.layer(0).unwrap();
    assert!(p.end_forward().is_err());
    p.begin_forward(false).unwrap();
    for l in 0..LAYERS {
        p.layer(l).unwrap();
    }
    p.end_forward().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn incremental_decoding_matches_processing_the_whole_prompt() {
    let dir = temp_dir("kv");
    write_tiny_model(&dir, 6);
    let tokens: Vec<u32> = (0..20).map(|i| (i * 31 + 7) % VOCAB as u32).collect();
    let mut whole = model_from(Box::new(ResidentProvider::load(plain(&dir)).unwrap()));
    let want = whole.forward(&tokens).unwrap();
    let mut inc = model_from(Box::new(ResidentProvider::load(plain(&dir)).unwrap()));
    inc.forward(&tokens[..9]).unwrap();
    for t in &tokens[9..19] {
        inc.forward(&[*t]).unwrap();
    }
    let got = inc.forward(&tokens[19..]).unwrap();
    let worst = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "cached decoding drifted by {worst}");
    assert_eq!(inc.pos(), 20);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn bad_requests_are_errors_not_panics() {
    let dir = temp_dir("bad");
    write_tiny_model(&dir, 7);
    let mut m = model_from(Box::new(ResidentProvider::load(plain(&dir)).unwrap()));
    assert!(m.forward(&[]).is_err());
    assert!(m.forward(&[VOCAB as u32]).is_err());
    assert!(m.forward(&vec![1u32; 65]).is_err(), "longer than max_ctx");
    m.forward(&vec![1u32; 60]).unwrap();
    assert!(m.forward(&[1u32; 5]).is_err(), "would overflow the cache");
    m.reset();
    assert!(m.forward(&vec![1u32; 64]).is_ok());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_storage_throttle_really_limits_read_speed() {
    let dir = temp_dir("throttle");
    write_tiny_model(&dir, 8);
    let files = ModelFiles::open(&dir).unwrap();
    let layer = files.layout.total;
    let rate_mb = 4.0; // MB/s
    let src = FileSource::open(files).unwrap().with_throttle(Some(Throttle::new(rate_mb)));
    let mut buf = vec![0u8; layer];
    let n = 40;
    let t = Instant::now();
    for i in 0..n {
        src.read_layer(i % LAYERS, &mut buf).unwrap();
    }
    let elapsed = t.elapsed().as_secs_f64();
    let expected = (n * layer) as f64 / (rate_mb * 1e6);
    assert!(elapsed >= expected * 0.95, "read {} bytes in {elapsed:.3}s, faster than the cap allows ({expected:.3}s)", n * layer);
    assert!(elapsed < expected * 1.10 + 0.05, "throttle delivers only {:.0}% of the asked rate: {elapsed:.3}s vs {expected:.3}s", 100.0 * expected / elapsed);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn streaming_holds_far_less_than_the_model() {
    let dir = temp_dir("report");
    write_tiny_model(&dir, 9);
    let r = ResidentProvider::load(plain(&dir)).unwrap().report();
    let s = StreamProvider::new(plain(&dir), 2, false).report();
    // Embedding table dominates this tiny model; streaming holds two layer-sized buffers only.
    assert!(s.resident_weight_bytes < r.resident_weight_bytes, "{} vs {}", s.resident_weight_bytes, r.resident_weight_bytes);
    std::fs::remove_dir_all(&dir).unwrap();
}
