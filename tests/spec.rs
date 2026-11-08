//! Speculative decoding, on a tiny synthetic model (no download needed).

mod common;

use common::*;
use layerstream::model::{Model, Sampler};
use layerstream::provider::{ResidentProvider, StreamProvider};
use layerstream::quant;
use layerstream::source::FileSource;
use layerstream::spec::speculative_greedy;
use layerstream::weights::{ModelFiles, WeightFormat};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const PROMPT: [u32; 6] = [5, 17, 301, 42, 599, 0];
const OTHER_PROMPT: [u32; 3] = [250, 3, 77];

fn plain(dir: &Path) -> Arc<FileSource> {
    Arc::new(FileSource::open(ModelFiles::open(dir).unwrap()).unwrap())
}

fn plain_quant(path: &Path) -> Arc<FileSource> {
    Arc::new(FileSource::open(ModelFiles::open_quantized(path).unwrap()).unwrap())
}

fn quantize(dir: &Path, name: &str, bits: u8) -> PathBuf {
    let out = dir.join(name);
    let _ = std::fs::remove_file(&out);
    let cfg = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let f = WeightFormat::Quant { bits, group: 32 };
    quant::quantize_model(&plain(dir), &cfg, f, f, true, 3, &out).unwrap();
    out
}

fn resident(src: Arc<FileSource>) -> Model {
    Model::new(Box::new(ResidentProvider::load(src).unwrap()), 3, 128)
}

fn streamed(src: Arc<FileSource>) -> Model {
    Model::new(Box::new(StreamProvider::new(src, 2, false)), 3, 128)
}

/// What the target alone produces with greedy decoding.
fn plain_greedy(m: &mut Model, prompt: &[u32], max_new: usize, eos: &[u32]) -> Vec<u32> {
    let mut s = Sampler::greedy();
    let mut logits = m.forward(prompt).unwrap();
    let mut out = Vec::new();
    for i in 0..max_new {
        let t = s.sample(&logits);
        out.push(t);
        if eos.contains(&t) || i + 1 == max_new {
            break;
        }
        logits = m.forward(&[t]).unwrap();
    }
    out
}

fn bit_equal(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()))
}

#[test]
fn checking_several_positions_at_once_gives_bit_identical_logits() {
    let dir = temp_dir("spec-multi");
    write_tiny_model(&dir, 11);
    let block = [8u32, 9, 400, 33, 120];
    for n in [1usize, 2, block.len()] {
        for stream in [false, true] {
            let make = |d: &Path| if stream { streamed(plain(d)) } else { resident(plain(d)) };
            let mut a = make(&dir);
            a.forward(&PROMPT).unwrap();
            let together = a.forward_multi(&block, n).unwrap();
            assert_eq!(together.len(), n);

            let mut b = make(&dir);
            b.forward(&PROMPT).unwrap();
            let one_by_one: Vec<Vec<f32>> = block.iter().map(|&t| b.forward(&[t]).unwrap()).collect();
            assert!(bit_equal(&together, &one_by_one[block.len() - n..]), "n={n} stream={stream}");
            assert_eq!(a.pos(), b.pos());
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn truncating_the_cache_forgets_the_tail_exactly() {
    let dir = temp_dir("spec-truncate");
    write_tiny_model(&dir, 12);
    let mut a = resident(plain(&dir));
    a.forward(&PROMPT).unwrap();
    a.forward(&[7, 8, 9]).unwrap();
    a.truncate(PROMPT.len());
    let after_truncate = a.forward(&[20]).unwrap();

    let mut b = resident(plain(&dir));
    b.forward(&PROMPT).unwrap();
    let never_saw_them = b.forward(&[20]).unwrap();
    assert!(bit_equal(&[after_truncate], &[never_saw_them]));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn speculation_never_changes_the_output_whatever_the_draft_guesses() {
    let dir = temp_dir("spec-same");
    write_tiny_model(&dir, 13);
    let other = temp_dir("spec-other");
    write_tiny_model(&other, 99); // an unrelated model: its guesses will mostly be wrong
    let q8 = quantize(&dir, "d8.lsq", 8);
    let q4 = quantize(&dir, "d4.lsq", 4);

    let drafts: Vec<(&str, Arc<FileSource>)> = vec![("int8 copy", plain_quant(&q8)), ("int4 copy", plain_quant(&q4)), ("unrelated model", plain(&other))];
    for (name, src) in &drafts {
        for prompt in [&PROMPT[..], &OTHER_PROMPT[..]] {
            for max_new in [1usize, 2, 9, 30] {
                let want = plain_greedy(&mut resident(plain(&dir)), prompt, max_new, &[]);
                for k in [1usize, 2, 3, 5, 8] {
                    for stream_target in [false, true] {
                        let mut target = if stream_target { streamed(plain(&dir)) } else { resident(plain(&dir)) };
                        let mut draft = resident(src.clone());
                        let got = speculative_greedy(&mut target, &mut draft, prompt, max_new, k, &[]).unwrap();
                        assert_eq!(got.tokens, want, "draft={name} prompt len {} max_new={max_new} k={k} stream={stream_target}", prompt.len());
                        assert!(got.stats.accepted <= got.stats.drafted);
                        assert!(got.stats.target_passes <= got.tokens.len().saturating_sub(1), "each pass must yield at least one token");
                    }
                }
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::remove_dir_all(&other).unwrap();
}

#[test]
fn it_stops_at_an_end_of_sequence_token_exactly_where_plain_greedy_does() {
    let dir = temp_dir("spec-eos");
    write_tiny_model(&dir, 14);
    let full = plain_greedy(&mut resident(plain(&dir)), &PROMPT, 24, &[]);
    for cut in [0usize, 1, 5, 11] {
        let eos = [full[cut]];
        let want = plain_greedy(&mut resident(plain(&dir)), &PROMPT, 24, &eos);
        assert!(want.len() <= cut + 1 && *want.last().unwrap() == eos[0]);
        for k in [1usize, 4, 8] {
            let mut target = resident(plain(&dir));
            let mut draft = resident(plain(&dir));
            let got = speculative_greedy(&mut target, &mut draft, &PROMPT, 24, k, &eos).unwrap();
            assert_eq!(got.tokens, want, "cut={cut} k={k}");
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_perfect_draft_needs_far_fewer_passes_over_the_weights() {
    let dir = temp_dir("spec-perfect");
    write_tiny_model(&dir, 15);
    let max_new = 21;
    let k = 3;

    let mut base = streamed(plain(&dir));
    let want = plain_greedy(&mut base, &PROMPT, max_new, &[]);
    let base_bytes = base.report().io.bytes_read;

    let mut target = streamed(plain(&dir));
    let mut draft = resident(plain(&dir)); // the same model, so every guess is right
    let got = speculative_greedy(&mut target, &mut draft, &PROMPT, max_new, k, &[]).unwrap();
    assert_eq!(got.tokens, want);
    assert_eq!(got.stats.accepted, got.stats.drafted, "a perfect draft is never wrong");
    assert_eq!(got.stats.target_passes, (max_new - 1).div_ceil(k + 1));
    let spec_bytes = target.report().io.bytes_read;
    assert!(spec_bytes * 3 < base_bytes, "{spec_bytes} bytes read with speculation, {base_bytes} without");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn bad_requests_are_errors_not_panics() {
    let dir = temp_dir("spec-bad");
    write_tiny_model(&dir, 16);
    let fresh = || (resident(plain(&dir)), resident(plain(&dir)));

    let (mut t, mut d) = fresh();
    for (k, prompt, max_new) in [(0usize, &PROMPT[..], 5usize), (33, &PROMPT[..], 5), (2, &[][..], 5), (2, &PROMPT[..], 0), (2, &PROMPT[..], 500)] {
        assert!(speculative_greedy(&mut t, &mut d, prompt, max_new, k, &[]).is_err(), "k={k} len={} max_new={max_new}", prompt.len());
    }
    // a model that has already been used is refused
    t.forward(&PROMPT).unwrap();
    assert!(speculative_greedy(&mut t, &mut d, &PROMPT, 5, 2, &[]).is_err());
    // asking for logits at more positions than there are tokens
    let (mut t, _) = fresh();
    assert!(t.forward_multi(&[1, 2], 3).is_err());
    assert!(t.forward_multi(&[1, 2], 0).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
