//! Shared helpers: generate a tiny synthetic Qwen2-shaped model on disk so the full pipeline
//! (loading, streaming, sealing, decryption, forward pass) can be tested without downloading
//! the real 1 GB model.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub const HIDDEN: usize = 64;
pub const HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const INTER: usize = 128;
pub const LAYERS: usize = 3;
/// Deliberately not a multiple of any chunk size used in the tests.
pub const VOCAB: usize = 600;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

fn bf16_bytes(vals: impl Iterator<Item = f32>) -> Vec<u8> {
    vals.flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect()
}

pub fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("layerstream-test-{}-{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// `(bits, group)` formats for a model whose weights sit exactly on a quantization grid.
#[derive(Clone, Copy)]
pub struct Grid {
    pub body: (u8, usize),
    pub embed: (u8, usize),
}

/// Weights that are exactly `code * 2^-k` with integer codes, and with the largest code in
/// every group pinned at the maximum. Such a model is representable in bf16 *and* in the
/// quantized format without any rounding, so the two must compute the same thing.
fn grid_bytes(rng: &mut Lcg, rows: usize, cols: usize, bits: u8, group: usize) -> Vec<u8> {
    let (qmax, scale) = if bits == 8 { (127i32, 2f32.powi(-10)) } else { (7i32, 2f32.powi(-6)) };
    let mut vals = Vec::with_capacity(rows * cols);
    for _ in 0..rows {
        for g in 0..cols / group {
            for j in 0..group {
                let mut code = ((rng.next() * (qmax as f32 + 0.5)).round() as i32).clamp(-qmax, qmax);
                if j == (g * 7) % group {
                    code = if rng.next() < 0.0 { -qmax } else { qmax };
                }
                vals.push(code as f32 * scale);
            }
        }
    }
    bf16_bytes(vals.into_iter())
}

/// Write `config.json` and `model.safetensors` for a tiny model into `dir`.
pub fn write_tiny_model(dir: &Path, seed: u64) {
    write_model(dir, seed, None);
}

/// Like [`write_tiny_model`], but every weight matrix lies exactly on the given grid.
pub fn write_tiny_grid_model(dir: &Path, seed: u64, grid: Grid) {
    write_model(dir, seed, Some(grid));
}

fn write_model(dir: &Path, seed: u64, grid: Option<Grid>) {
    let kv = KV_HEADS * (HIDDEN / HEADS);
    let config = format!(
        r#"{{"model_type":"qwen2","hidden_act":"silu","tie_word_embeddings":true,"hidden_size":{HIDDEN},
"num_attention_heads":{HEADS},"num_key_value_heads":{KV_HEADS},"intermediate_size":{INTER},
"num_hidden_layers":{LAYERS},"vocab_size":{VOCAB},"rms_norm_eps":1e-6,"rope_theta":10000.0,
"max_position_embeddings":512}}"#
    );
    std::fs::write(dir.join("config.json"), config).unwrap();

    // Tensors in alphabetical order, like the real file.
    let mut tensors: Vec<(String, Vec<usize>)> = vec![("model.embed_tokens.weight".into(), vec![VOCAB, HIDDEN])];
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![HIDDEN]),
            ("mlp.down_proj.weight", vec![HIDDEN, INTER]),
            ("mlp.gate_proj.weight", vec![INTER, HIDDEN]),
            ("mlp.up_proj.weight", vec![INTER, HIDDEN]),
            ("post_attention_layernorm.weight", vec![HIDDEN]),
            ("self_attn.k_proj.bias", vec![kv]),
            ("self_attn.k_proj.weight", vec![kv, HIDDEN]),
            ("self_attn.o_proj.weight", vec![HIDDEN, HIDDEN]),
            ("self_attn.q_proj.bias", vec![HIDDEN]),
            ("self_attn.q_proj.weight", vec![HIDDEN, HIDDEN]),
            ("self_attn.v_proj.bias", vec![kv]),
            ("self_attn.v_proj.weight", vec![kv, HIDDEN]),
        ] {
            tensors.push((format!("{p}.{suffix}"), shape));
        }
    }
    tensors.push(("model.norm.weight".into(), vec![HIDDEN]));

    let mut rng = Lcg(seed);
    let mut data = Vec::new();
    let mut entries = Vec::new();
    for (name, shape) in &tensors {
        let n: usize = shape.iter().product();
        let is_norm = name.ends_with("layernorm.weight") || name == "model.norm.weight";
        let bytes = if is_norm {
            bf16_bytes((0..n).map(|_| 1.0 + 0.1 * rng.next()))
        } else if let (Some(g), 2) = (grid, shape.len()) {
            let (bits, group) = if name == "model.embed_tokens.weight" { g.embed } else { g.body };
            grid_bytes(&mut rng, shape[0], shape[1], bits, group)
        } else {
            bf16_bytes((0..n).map(|_| 0.15 * rng.next()))
        };
        let (a, b) = (data.len(), data.len() + bytes.len());
        data.extend_from_slice(&bytes);
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        entries.push(format!(r#""{name}":{{"dtype":"BF16","shape":[{}],"data_offsets":[{a},{b}]}}"#, dims.join(",")));
    }
    let mut header = format!(r#"{{"__metadata__":{{"format":"pt"}},{}}}"#, entries.join(","));
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut file = Vec::new();
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
}
