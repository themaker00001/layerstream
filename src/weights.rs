//! Model configuration and where each tensor lives.
//!
//! A transformer layer is stored as 12 tensors that sit next to each other in the weights
//! file. This module works out, once, the layout of those 12 tensors inside one "layer
//! block" (the same for every layer), so a layer can be fetched with a single read and then
//! viewed tensor by tensor.

use crate::json::Json;
use crate::safetensors::Header;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub hidden: usize,
    pub inter: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub max_pos: usize,
}

impl Config {
    pub fn from_json(j: &Json) -> Result<Config, String> {
        let u = |k: &str| -> Result<usize, String> {
            j.get(k).and_then(Json::as_u64).map(|v| v as usize).ok_or_else(|| format!("config.json: missing integer {k}"))
        };
        let f = |k: &str| -> Result<f32, String> {
            j.get(k).and_then(Json::as_f64).map(|v| v as f32).ok_or_else(|| format!("config.json: missing number {k}"))
        };
        if j.get("model_type").and_then(Json::as_str) != Some("qwen2") {
            return Err("only Qwen2-architecture models are supported".into());
        }
        if j.get("hidden_act").and_then(Json::as_str) != Some("silu") {
            return Err("only SiLU activations are supported".into());
        }
        if j.get("tie_word_embeddings").and_then(Json::as_bool) != Some(true) {
            return Err("only models with tied input/output embeddings are supported".into());
        }
        let hidden = u("hidden_size")?;
        let heads = u("num_attention_heads")?;
        let kv_heads = u("num_key_value_heads")?;
        if hidden % heads != 0 || heads % kv_heads != 0 {
            return Err("inconsistent head configuration".into());
        }
        let head_dim = hidden / heads;
        if head_dim % 2 != 0 {
            return Err("head dimension must be even for rotary embeddings".into());
        }
        Ok(Config {
            hidden,
            inter: u("intermediate_size")?,
            layers: u("num_hidden_layers")?,
            heads,
            kv_heads,
            head_dim,
            vocab: u("vocab_size")?,
            eps: f("rms_norm_eps")?,
            rope_theta: f("rope_theta")?,
            max_pos: u("max_position_embeddings")?,
        })
    }

    pub fn load(dir: &Path) -> Result<Config, String> {
        let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?;
        Config::from_json(&Json::parse(&text)?)
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Bytes of one embedding row in bf16.
    pub fn row_bytes(&self) -> usize {
        self.hidden * 2
    }
}

// Tensor slots inside a layer block, in the order they are stored in the weights file.
pub const T_INPUT_NORM: usize = 0;
pub const T_DOWN: usize = 1;
pub const T_GATE: usize = 2;
pub const T_UP: usize = 3;
pub const T_POST_NORM: usize = 4;
pub const T_K_B: usize = 5;
pub const T_K_W: usize = 6;
pub const T_O_W: usize = 7;
pub const T_Q_B: usize = 8;
pub const T_Q_W: usize = 9;
pub const T_V_B: usize = 10;
pub const T_V_W: usize = 11;
const N_TENSORS: usize = 12;

const SUFFIX: [&str; N_TENSORS] = [
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.v_proj.weight",
];

fn expected_shape(slot: usize, c: &Config) -> Vec<usize> {
    let (h, i, kv) = (c.hidden, c.inter, c.kv_dim());
    match slot {
        T_INPUT_NORM | T_POST_NORM | T_Q_B => vec![h],
        T_K_B | T_V_B => vec![kv],
        T_DOWN => vec![h, i],
        T_GATE | T_UP => vec![i, h],
        T_K_W | T_V_W => vec![kv, h],
        T_O_W | T_Q_W => vec![h, h],
        _ => unreachable!(),
    }
}

/// Where each of the 12 tensors sits inside a layer block: `(offset, length)` in bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerLayout {
    pub ranges: [(usize, usize); N_TENSORS],
    pub total: usize,
}

/// A layer block's bytes, viewed tensor by tensor.
pub struct LayerView<'a> {
    buf: &'a [u8],
    layout: &'a LayerLayout,
}

impl<'a> LayerView<'a> {
    pub fn new(buf: &'a [u8], layout: &'a LayerLayout) -> Self {
        assert_eq!(buf.len(), layout.total, "layer block has the wrong size");
        LayerView { buf, layout }
    }

    pub fn t(&self, slot: usize) -> &'a [u8] {
        let (o, l) = self.layout.ranges[slot];
        &self.buf[o..o + l]
    }
}

/// Everything known about a model on disk, without having loaded its weights.
pub struct ModelFiles {
    pub cfg: Config,
    pub path: PathBuf,
    pub layout: LayerLayout,
    /// Absolute file offset of each layer block.
    pub layer_start: Vec<u64>,
    /// Absolute file offset and length of the embedding table.
    pub embed_start: u64,
    pub embed_len: usize,
    /// Absolute file offset and length of the final norm tensor.
    pub norm_start: u64,
    pub norm_len: usize,
    pub final_norm: Vec<f32>,
}

impl ModelFiles {
    pub fn open(dir: &Path) -> Result<ModelFiles, String> {
        let cfg = Config::load(dir)?;
        let path = dir.join("model.safetensors");
        let header = Header::read(&path)?;

        let embed = header.bf16_tensor("model.embed_tokens.weight", &[cfg.vocab, cfg.hidden])?;
        let norm = header.bf16_tensor("model.norm.weight", &[cfg.hidden])?;

        let mut layout: Option<LayerLayout> = None;
        let mut layer_start = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let mut items: Vec<(usize, u64, u64)> = Vec::new(); // slot, start, end
            for (slot, suffix) in SUFFIX.iter().enumerate() {
                let t = header.bf16_tensor(&format!("model.layers.{l}.{suffix}"), &expected_shape(slot, &cfg))?;
                items.push((slot, t.start, t.end));
            }
            items.sort_by_key(|&(_, s, _)| s);
            if items.windows(2).any(|w| w[0].2 != w[1].1) {
                return Err(format!("layer {l}: tensors are not contiguous in the file"));
            }
            let span_start = items[0].1;
            let span_end = items[N_TENSORS - 1].2;
            let mut ranges = [(0usize, 0usize); N_TENSORS];
            for (slot, s, e) in &items {
                ranges[*slot] = ((*s - span_start) as usize, (*e - *s) as usize);
            }
            let this = LayerLayout { ranges, total: (span_end - span_start) as usize };
            match &layout {
                None => layout = Some(this),
                Some(first) if *first == this => {}
                Some(_) => return Err(format!("layer {l} is laid out differently from layer 0")),
            }
            layer_start.push(span_start);
        }

        // The final norm is tiny, so it stays resident as f32.
        let file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
        let mut raw = vec![0u8; norm.len()];
        std::os::unix::fs::FileExt::read_exact_at(&file, &mut raw, norm.start).map_err(|e| e.to_string())?;
        let mut final_norm = vec![0.0f32; cfg.hidden];
        crate::kernels::bf16_slice_to_f32(&raw, &mut final_norm);

        Ok(ModelFiles {
            cfg,
            path,
            layout: layout.ok_or("model has no layers")?,
            layer_start,
            embed_start: embed.start,
            embed_len: embed.len(),
            norm_start: norm.start,
            norm_len: norm.len(),
            final_norm,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_unsupported_architectures() {
        let base = r#"{"model_type":"qwen2","hidden_act":"silu","tie_word_embeddings":true,"hidden_size":896,
            "num_attention_heads":14,"num_key_value_heads":2,"intermediate_size":4864,"num_hidden_layers":24,
            "vocab_size":151936,"rms_norm_eps":1e-6,"rope_theta":1000000.0,"max_position_embeddings":32768}"#;
        let c = Config::from_json(&Json::parse(base).unwrap()).unwrap();
        assert_eq!((c.head_dim, c.kv_dim(), c.row_bytes()), (64, 128, 1792));
        for (from, to) in [
            ("\"qwen2\"", "\"llama\""),
            ("\"silu\"", "\"gelu\""),
            ("\"tie_word_embeddings\":true", "\"tie_word_embeddings\":false"),
            ("\"num_key_value_heads\":2", "\"num_key_value_heads\":5"),
        ] {
            let bad = base.replace(from, to);
            assert!(Config::from_json(&Json::parse(&bad).unwrap()).is_err(), "accepted {to}");
        }
    }
}
