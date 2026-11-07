//! End-to-end checks against the reference implementation (tests/golden.json) and between
//! the engine's own modes. Needs the model in models/Qwen2.5-0.5B-Instruct; skips (loudly)
//! if it is absent.

use layerstream::json::Json;
use layerstream::model::{Model, Sampler};
use layerstream::provider::{Provider, ResidentProvider, StreamProvider};
use layerstream::source::FileSource;
use layerstream::weights::ModelFiles;
use std::path::PathBuf;
use std::sync::Arc;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model_dir() -> PathBuf {
    root().join("models/Qwen2.5-0.5B-Instruct")
}

fn have_model() -> bool {
    let ok = model_dir().join("model.safetensors").exists();
    if !ok {
        eprintln!("SKIPPED: model files not downloaded (see README)");
    }
    ok
}

fn source() -> Arc<FileSource> {
    Arc::new(FileSource::open(ModelFiles::open(&model_dir()).unwrap()).unwrap())
}

fn resident() -> Model {
    Model::new(Box::new(ResidentProvider::load(source()).unwrap()), 4, 256)
}

fn streamed(slots: usize) -> Model {
    let p: Box<dyn Provider> = Box::new(StreamProvider::new(source(), slots, false));
    Model::new(p, 4, 256)
}

struct Prompt {
    ids: Vec<u32>,
    top10: Vec<(usize, f32)>,
    probes: Vec<(usize, f32)>,
    greedy: Vec<u32>,
}

fn golden_prompts() -> Vec<Prompt> {
    let text = std::fs::read_to_string(root().join("tests/golden.json")).unwrap();
    let g = Json::parse(&text).unwrap();
    let pairs = |v: &Json| -> Vec<(usize, f32)> {
        v.as_arr()
            .unwrap()
            .iter()
            .map(|e| (e.get("id").unwrap().as_u64().unwrap() as usize, e.get("logit").unwrap().as_f64().unwrap() as f32))
            .collect()
    };
    let ids = |v: &Json| -> Vec<u32> { v.as_arr().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect() };
    g.get("prompts")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|p| Prompt {
            ids: ids(p.get("ids").unwrap()),
            top10: pairs(p.get("top10").unwrap()),
            probes: pairs(p.get("probes").unwrap()),
            greedy: ids(p.get("greedy_ids").unwrap()),
        })
        .collect()
}

const EOS: [u32; 2] = [151645, 151643];

fn generate(model: &mut Model, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut sampler = Sampler::greedy();
    let mut out = Vec::new();
    let mut logits = model.forward(prompt).unwrap();
    for _ in 0..n {
        let next = sampler.sample(&logits);
        out.push(next);
        if EOS.contains(&next) {
            break;
        }
        logits = model.forward(&[next]).unwrap();
    }
    out
}

#[test]
fn logits_match_the_reference_implementation() {
    if !have_model() {
        return;
    }
    let mut worst = 0.0f32;
    for p in golden_prompts() {
        // Fresh model per prompt so each starts with an empty cache.
        let mut m = resident();
        let logits = m.forward(&p.ids).unwrap();
        assert_eq!(logits.len(), m.cfg().vocab);
        for (id, want) in p.top10.iter().chain(p.probes.iter()) {
            let d = (logits[*id] - want).abs();
            worst = worst.max(d);
            assert!(d < 0.05, "logit {id}: got {}, reference {want} (diff {d})", logits[*id]);
        }
        let mut ours: Vec<usize> = (0..logits.len()).collect();
        ours.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        assert_eq!(ours[0], p.top10[0].0, "top-1 token differs from the reference");
        let want_set: std::collections::HashSet<usize> = p.top10.iter().map(|(i, _)| *i).collect();
        let got_set: std::collections::HashSet<usize> = ours[..10].iter().copied().collect();
        assert_eq!(want_set, got_set, "top-10 set differs from the reference");
    }
    eprintln!("largest logit difference from the reference: {worst:.2e}");
}

#[test]
fn greedy_generation_matches_the_reference() {
    if !have_model() {
        return;
    }
    for p in golden_prompts() {
        let mut m = resident();
        let got = generate(&mut m, &p.ids, p.greedy.len());
        assert_eq!(got, p.greedy, "generated tokens differ from the reference");
    }
}

#[test]
fn streaming_is_bit_identical_to_resident() {
    if !have_model() {
        return;
    }
    let p = &golden_prompts()[0];
    let mut a = resident();
    let la = a.forward(&p.ids).unwrap();
    for slots in [1usize, 2, 3] {
        let mut b = streamed(slots);
        let lb = b.forward(&p.ids).unwrap();
        assert!(la.iter().zip(&lb).all(|(x, y)| x.to_bits() == y.to_bits()), "prefill differs with {slots} slots");
    }
    // And through several single-token decode steps, which reuse the cache.
    let mut a = resident();
    let mut b = streamed(2);
    let mut la = a.forward(&p.ids).unwrap();
    let mut lb = b.forward(&p.ids).unwrap();
    let mut s = Sampler::greedy();
    for step in 0..4 {
        assert!(la.iter().zip(&lb).all(|(x, y)| x.to_bits() == y.to_bits()), "decode step {step} differs");
        let next = s.sample(&la);
        la = a.forward(&[next]).unwrap();
        lb = b.forward(&[next]).unwrap();
    }
}

#[test]
fn prefill_in_pieces_equals_prefill_at_once() {
    if !have_model() {
        return;
    }
    let p = &golden_prompts()[0];
    let mut once = resident();
    let l1 = once.forward(&p.ids).unwrap();
    let mut pieces = resident();
    let mid = p.ids.len() / 2;
    pieces.forward(&p.ids[..mid]).unwrap();
    let l2 = pieces.forward(&p.ids[mid..]).unwrap();
    let worst = l1.iter().zip(&l2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1e-3, "cache handling differs: {worst}");
    assert_eq!(pieces.pos(), p.ids.len());
}

#[test]
fn rejects_bad_requests() {
    if !have_model() {
        return;
    }
    let mut m = resident();
    assert!(m.forward(&[]).is_err());
    assert!(m.forward(&[151936]).is_err());
    let too_many = vec![1u32; 257];
    assert!(m.forward(&too_many).is_err());
}
