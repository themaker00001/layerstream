//! Quantized models, end to end, on a tiny synthetic model (no download needed).

mod common;

use common::*;
use layerstream::model::{Model, Sampler};
use layerstream::provider::{Provider, ResidentProvider, StreamProvider};
use layerstream::quant;
use layerstream::seal::{self, OpenOptions, SealParams, SealedSource, SecretKey};
use layerstream::source::{FileSource, Source};
use layerstream::weights::{ModelFiles, QuantHeader, WeightFormat};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn fmt((bits, group): (u8, usize)) -> WeightFormat {
    WeightFormat::Quant { bits, group }
}

fn plain(dir: &Path) -> Arc<FileSource> {
    Arc::new(FileSource::open(ModelFiles::open(dir).unwrap()).unwrap())
}

fn plain_quant(path: &Path) -> Arc<FileSource> {
    Arc::new(FileSource::open(ModelFiles::open_quantized(path).unwrap()).unwrap())
}

fn quantize(dir: &Path, name: &str, body: (u8, usize), embed: (u8, usize), clip: bool) -> PathBuf {
    let out = dir.join(name);
    let _ = std::fs::remove_file(&out);
    let cfg = std::fs::read_to_string(dir.join("config.json")).unwrap();
    quant::quantize_model(&plain(dir), &cfg, fmt(body), fmt(embed), clip, 3, &out).unwrap();
    out
}

fn model_from(p: Box<dyn Provider>) -> Model {
    Model::new(p, 3, 64)
}

const PROMPT: [u32; 6] = [5, 17, 301, 42, 599, 0];
const FOLLOW: [u32; 3] = [7, 33, 120];

/// Logits after the prompt and after each of a few fixed follow-up tokens (so the result does
/// not depend on which token a slightly different model would pick).
fn run_fixed(m: &mut Model) -> Vec<Vec<f32>> {
    let mut out = vec![m.forward(&PROMPT).unwrap()];
    for t in FOLLOW {
        out.push(m.forward(&[t]).unwrap());
    }
    out
}

fn run_greedy(m: &mut Model) -> Vec<Vec<f32>> {
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

fn max_abs(a: &[Vec<f32>]) -> f32 {
    a.iter().flatten().fold(0.0, |m, v| m.max(v.abs()))
}

fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).flat_map(|(x, y)| x.iter().zip(y)).fold(0.0, |m, (p, q)| m.max((p - q).abs()))
}

#[test]
fn a_quantized_model_computes_the_same_thing_as_bf16_when_the_weights_fit_the_grid() {
    // If every weight is exactly representable in both formats, the only difference left is
    // the order of floating-point additions. Any layout, nibble-order or scale bug would show
    // up as a large difference.
    let cases = [
        Grid { body: (8, 32), embed: (8, 32) },
        Grid { body: (8, 16), embed: (8, 64) },
        Grid { body: (4, 32), embed: (4, 32) },
        Grid { body: (4, 16), embed: (4, 64) },
        Grid { body: (4, 32), embed: (8, 16) }, // mixed: 4-bit layers, 8-bit embedding table
        Grid { body: (8, 64), embed: (4, 16) },
    ];
    for (i, grid) in cases.into_iter().enumerate() {
        let dir = temp_dir(&format!("grid{i}"));
        write_tiny_grid_model(&dir, 100 + i as u64, grid);
        let reference = run_fixed(&mut model_from(Box::new(ResidentProvider::load(plain(&dir)).unwrap())));
        assert!(max_abs(&reference) > 0.01, "case {i}: the reference logits are degenerate");

        for clip in [false, true] {
            let q = quantize(&dir, "m.lsq", grid.body, grid.embed, clip);
            let got = run_fixed(&mut model_from(Box::new(ResidentProvider::load(plain_quant(&q)).unwrap())));
            let tol = 1e-3 * (1.0 + max_abs(&reference));
            let d = max_diff(&reference, &got);
            assert!(d < tol, "case {i} (body {:?}, embed {:?}, clip={clip}): max logit difference {d} (tolerance {tol})", grid.body, grid.embed);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn rounding_real_weights_stays_close_and_eight_bits_is_closer_than_four() {
    let dir = temp_dir("random");
    write_tiny_model(&dir, 7);
    let reference = run_fixed(&mut model_from(Box::new(ResidentProvider::load(plain(&dir)).unwrap())));
    let scale = max_abs(&reference);
    let err = |body: (u8, usize), embed: (u8, usize)| {
        let q = quantize(&dir, "r.lsq", body, embed, true);
        let got = run_fixed(&mut model_from(Box::new(ResidentProvider::load(plain_quant(&q)).unwrap())));
        assert!(got.iter().flatten().all(|v| v.is_finite()));
        max_diff(&reference, &got) / scale
    };
    let (e8, e4) = (err((8, 32), (8, 32)), err((4, 32), (4, 32)));
    assert!(e8 < 0.05, "8-bit error {e8} is larger than expected");
    assert!(e4 > e8, "4-bit ({e4}) should be coarser than 8-bit ({e8})");
    assert!(e4 < 1.0, "4-bit error {e4} is wild");
    std::fs::remove_dir_all(&dir).unwrap();
}

fn key(n: u8) -> SecretKey {
    SecretKey::from_bytes([n; 32])
}

const SMALL: SealParams = SealParams { layer_chunk: 700, embed_chunk_rows: 37 };

fn seal_quant(qpath: &Path, k: &SecretKey) -> PathBuf {
    let out = qpath.with_extension("sealed");
    let _ = std::fs::remove_file(&out);
    let header = QuantHeader::read(qpath).unwrap();
    seal::seal_model(&plain_quant(qpath), &header.config_text, k, *b"abcdefgh", SMALL, &out).unwrap();
    out
}

fn sealed(path: &Path, k: SecretKey) -> Arc<SealedSource> {
    Arc::new(SealedSource::open(path, k, OpenOptions { threads: Some(3), ..Default::default() }).unwrap())
}

#[test]
fn every_mode_of_a_quantized_model_gives_bit_identical_logits() {
    let dir = temp_dir("q-modes");
    write_tiny_model(&dir, 3);
    for (body, embed) in [((8, 32), (8, 32)), ((4, 32), (8, 64)), ((4, 16), (4, 16))] {
        let q = quantize(&dir, "m.lsq", body, embed, true);
        let sealed_path = seal_quant(&q, &key(9));

        let reference = run_greedy(&mut model_from(Box::new(ResidentProvider::load(plain_quant(&q)).unwrap())));
        assert!(reference.iter().flatten().all(|v| v.is_finite()));

        for slots in [1usize, 2, 3] {
            for wipe in [false, true] {
                let m = &mut model_from(Box::new(StreamProvider::new(plain_quant(&q), slots, wipe)));
                assert!(bit_equal(&reference, &run_greedy(m)), "{body:?}/{embed:?}: stream, {slots} slots, wipe={wipe}");
            }
        }
        // The encrypted copy holds exactly the same bytes and gives exactly the same answers.
        seal::verify_same(plain_quant(&q).as_ref(), sealed(&sealed_path, key(9)).as_ref()).unwrap();
        let m = &mut model_from(Box::new(ResidentProvider::load(sealed(&sealed_path, key(9))).unwrap()));
        assert!(bit_equal(&reference, &run_greedy(m)), "{body:?}/{embed:?}: sealed resident");
        for slots in [1usize, 2] {
            let m = &mut model_from(Box::new(StreamProvider::new(sealed(&sealed_path, key(9)), slots, true)));
            assert!(bit_equal(&reference, &run_greedy(m)), "{body:?}/{embed:?}: sealed stream, {slots} slots");
        }
        // A wrong key is refused even though the config inside carries a quantized format.
        let err = SealedSource::open(&sealed_path, key(8), OpenOptions::default()).err().expect("wrong key must fail");
        assert!(err.contains("authentication failed"), "{err}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn quantized_files_are_smaller_and_the_sizes_add_up() {
    let dir = temp_dir("q-size");
    write_tiny_model(&dir, 4);
    let bf16_len = std::fs::metadata(dir.join("model.safetensors")).unwrap().len();
    let mut last = u64::MAX;
    for (body, embed) in [((8, 32), (8, 32)), ((4, 32), (8, 32)), ((4, 32), (4, 32))] {
        let q = quantize(&dir, "s.lsq", body, embed, false);
        let len = std::fs::metadata(&q).unwrap().len();
        assert!(len < bf16_len, "{body:?}/{embed:?}: {len} is not smaller than {bf16_len}");
        assert!(len < last, "{body:?}/{embed:?} should be smaller than the previous, more precise variant");
        last = len;
        let files = ModelFiles::open_quantized(&q).unwrap();
        assert_eq!(files.cfg.format, fmt(body));
        assert_eq!(files.cfg.embed_format, fmt(embed));
        assert_eq!(files.layout.total, layerstream::weights::layer_layout_for(&files.cfg).total);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn bad_quantization_requests_and_damaged_files_are_errors() {
    let dir = temp_dir("q-bad");
    write_tiny_model(&dir, 5);
    let cfg = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let out = dir.join("x.lsq");

    // A group that does not divide the layer widths; a bit width that does not exist.
    for bad in [(8u8, 48usize), (6, 32), (8, 512)] {
        let _ = std::fs::remove_file(&out);
        assert!(quant::quantize_model(&plain(&dir), &cfg, fmt(bad), fmt(bad), true, 2, &out).is_err(), "{bad:?}");
    }
    // The output file must not already exist (nothing is overwritten).
    let good = quantize(&dir, "good.lsq", (8, 32), (8, 32), true);
    let err = quant::quantize_model(&plain(&dir), &cfg, fmt((8, 32)), fmt((8, 32)), true, 2, &good).unwrap_err();
    assert!(err.contains("exist"), "{err}");
    // A quantized model cannot be quantized again, and a bf16 file is not a quantized one.
    let err = quant::quantize_model(&plain_quant(&good), &cfg, fmt((8, 32)), fmt((8, 32)), true, 2, &dir.join("again.lsq")).unwrap_err();
    assert!(err.contains("already quantized"), "{err}");
    assert!(ModelFiles::open_quantized(&dir.join("model.safetensors")).is_err());

    let bytes = std::fs::read(&good).unwrap();
    let write = |name: &str, data: &[u8]| {
        let p = dir.join(name);
        std::fs::write(&p, data).unwrap();
        p
    };
    assert!(ModelFiles::open_quantized(&write("trunc.lsq", &bytes[..bytes.len() - 3])).is_err(), "truncated file accepted");
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(ModelFiles::open_quantized(&write("long.lsq", &longer)).is_err(), "file with trailing bytes accepted");
    let mut magic = bytes.clone();
    magic[0] ^= 1;
    assert!(ModelFiles::open_quantized(&write("magic.lsq", &magic)).is_err(), "bad magic accepted");
    assert!(ModelFiles::open_quantized(&write("tiny.lsq", &bytes[..5])).is_err(), "5-byte file accepted");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_streamed_quantized_model_holds_far_less_than_the_model() {
    let dir = temp_dir("q-mem");
    write_tiny_model(&dir, 6);
    let q = quantize(&dir, "m.lsq", (4, 32), (4, 32), true);
    let src = plain_quant(&q);
    let total_file = std::fs::metadata(&q).unwrap().len();
    let mut m = model_from(Box::new(StreamProvider::new(src.clone(), 2, false)));
    run_greedy(&mut m);
    let rep = m.report();
    assert!(rep.resident_weight_bytes < total_file, "streaming held {} of {} bytes", rep.resident_weight_bytes, total_file);
    // Every layer plus the whole embedding table is read for every forward pass (4 here), and
    // one embedding row for each of the 9 input tokens (6 prompt tokens + 3 generated ones).
    let per_pass = (src.cfg().layers * src.layout().total + src.cfg().vocab * src.cfg().row_bytes()) as u64;
    assert_eq!(rep.io.bytes_read, 4 * per_pass + 9 * src.cfg().row_bytes() as u64);
    std::fs::remove_dir_all(&dir).unwrap();
}
