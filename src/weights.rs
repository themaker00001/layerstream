//! Model configuration and where each tensor lives.
//!
//! A transformer layer is stored as 12 tensors that sit next to each other in the weights
//! file. This module works out, once, the layout of those 12 tensors inside one "layer
//! block" (the same for every layer), so a layer can be fetched with a single read and then
//! viewed tensor by tensor.

use crate::json::Json;
use crate::safetensors::Header;
use std::path::{Path, PathBuf};

/// How the matrix weights (the seven projection matrices of every layer, and the embedding
/// table) are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// As shipped: 2 bytes per weight.
    Bf16,
    /// Symmetric integers, `bits` per weight (8 or 4), with one bf16 scale shared by every
    /// `group` consecutive weights of a row.
    Quant { bits: u8, group: usize },
}

impl WeightFormat {
    /// Bytes of one stored row of `cols` weights. A quantized row is its packed integers
    /// followed by its scales, so every row is self-contained.
    pub fn row_bytes(self, cols: usize) -> usize {
        match self {
            WeightFormat::Bf16 => cols * 2,
            WeightFormat::Quant { bits, group } => cols * bits as usize / 8 + cols / group * 2,
        }
    }

    /// Check the format against the column counts of every matrix it will be applied to.
    pub fn validate(self, cols: &[usize]) -> Result<(), String> {
        if let WeightFormat::Quant { bits, group } = self {
            if bits != 8 && bits != 4 {
                return Err(format!("quantization supports 8 or 4 bits, not {bits}"));
            }
            if !(16..=256).contains(&group) || !group.is_multiple_of(16) {
                return Err(format!("group size must be a multiple of 16 between 16 and 256, not {group}"));
            }
            if let Some(c) = cols.iter().find(|&&c| !c.is_multiple_of(group)) {
                return Err(format!("group size {group} does not divide a matrix width of {c}"));
            }
        }
        Ok(())
    }

    pub fn name(self) -> String {
        match self {
            WeightFormat::Bf16 => "bf16".into(),
            WeightFormat::Quant { bits, group } => format!("int{bits}, groups of {group}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Format of the seven matrices in every layer.
    pub format: WeightFormat,
    /// Format of the embedding table, which doubles as the output layer. It may be more
    /// precise than the layers: its rounding noise goes straight into the logits.
    pub embed_format: WeightFormat,
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
        let inter = u("intermediate_size")?;
        // Quantized files embed their format in the config text they carry.
        let (format, embed_format) = match j.get("layerstream_quant") {
            None => (WeightFormat::Bf16, WeightFormat::Bf16),
            Some(q) => {
                let get = |k: &str| q.get(k).and_then(Json::as_u64).ok_or_else(|| format!("config: layerstream_quant is missing {k}"));
                let quant = |bk: &str, gk: &str| -> Result<WeightFormat, String> {
                    let (bits, group) = (get(bk)?, get(gk)?);
                    if bits > 255 || group > 4096 {
                        return Err("config: implausible layerstream_quant values".into());
                    }
                    Ok(WeightFormat::Quant { bits: bits as u8, group: group as usize })
                };
                let body = quant("bits", "group")?;
                let embed = match (q.get("embed_bits"), q.get("embed_group")) {
                    (None, None) => body,
                    (Some(_), Some(_)) => quant("embed_bits", "embed_group")?,
                    _ => return Err("config: layerstream_quant needs embed_bits and embed_group together".into()),
                };
                (body, embed)
            }
        };
        format.validate(&[hidden, inter])?;
        embed_format.validate(&[hidden])?;
        Ok(Config {
            format,
            embed_format,
            hidden,
            inter,
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

    /// Bytes of one stored embedding row (in the embedding table's format).
    pub fn row_bytes(&self) -> usize {
        self.embed_format.row_bytes(self.hidden)
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
pub const N_TENSORS: usize = 12;

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

/// `(rows, cols)` of a slot if it is one of the seven weight matrices (the rest are norms and
/// biases, which are always bf16).
pub fn matrix_shape(slot: usize, c: &Config) -> Option<(usize, usize)> {
    match slot {
        T_DOWN | T_GATE | T_UP | T_K_W | T_O_W | T_Q_W | T_V_W => {
            let s = expected_shape(slot, c);
            Some((s[0], s[1]))
        }
        _ => None,
    }
}

/// Layer block layout used by quantized files: the same 12 tensors in slot order, with the
/// seven matrices in the config's weight format.
pub fn layer_layout_for(c: &Config) -> LayerLayout {
    let mut ranges = [(0usize, 0usize); N_TENSORS];
    let mut off = 0;
    for (slot, range) in ranges.iter_mut().enumerate() {
        let len = match matrix_shape(slot, c) {
            Some((rows, cols)) => rows * c.format.row_bytes(cols),
            None => expected_shape(slot, c)[0] * 2,
        };
        *range = (off, len);
        off += len;
    }
    LayerLayout { ranges, total: off }
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

/// Header of a quantized model file (`.lsq`): the config text (with the weight format
/// embedded) and the byte size of each part. Layers, the embedding table and the final norm
/// follow the header, in that order.
#[derive(Debug, Clone)]
pub struct QuantHeader {
    pub config_text: String,
    pub layer_bytes: usize,
    pub embed_bytes: usize,
    pub norm_bytes: usize,
    /// Absolute offset of the first layer block (set by [`QuantHeader::read`]).
    pub data_start: u64,
}

const QUANT_MAGIC: &[u8; 4] = b"LSQ1";
const QUANT_FORMAT: &str = "layerstream-quant-1";

impl QuantHeader {
    /// The bytes written at the start of the file: magic, header length, header JSON.
    pub fn to_bytes(&self) -> Vec<u8> {
        let json = format!(
            "{{\"format\":\"{QUANT_FORMAT}\",\"config_json\":{},\"layer_bytes\":{},\"embed_bytes\":{},\"norm_bytes\":{}}}",
            Json::Str(self.config_text.clone()).to_text(),
            self.layer_bytes,
            self.embed_bytes,
            self.norm_bytes
        );
        let mut out = Vec::with_capacity(8 + json.len());
        out.extend_from_slice(QUANT_MAGIC);
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(json.as_bytes());
        out
    }

    pub fn read(path: &Path) -> Result<QuantHeader, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let at = |buf: &mut [u8], off: u64| std::os::unix::fs::FileExt::read_exact_at(&file, buf, off);
        let mut frame = [0u8; 8];
        at(&mut frame, 0).map_err(|_| "file too short to be a quantized model".to_string())?;
        if &frame[..4] != QUANT_MAGIC {
            return Err("not a quantized layerstream model (bad magic bytes)".into());
        }
        let len = u32::from_le_bytes(frame[4..8].try_into().unwrap()) as usize;
        if len == 0 || len > (1 << 20) {
            return Err("implausible quantized header length".into());
        }
        let mut json = vec![0u8; len];
        at(&mut json, 8).map_err(|e| format!("quantized header: {e}"))?;
        let j = Json::parse(std::str::from_utf8(&json).map_err(|_| "quantized header is not UTF-8")?)?;
        if j.get("format").and_then(Json::as_str) != Some(QUANT_FORMAT) {
            return Err("unknown quantized file format".into());
        }
        let u = |k: &str| j.get(k).and_then(Json::as_u64).map(|v| v as usize).ok_or_else(|| format!("quantized header: missing {k}"));
        Ok(QuantHeader {
            config_text: j.get("config_json").and_then(Json::as_str).ok_or("quantized header: missing config")?.to_string(),
            layer_bytes: u("layer_bytes")?,
            embed_bytes: u("embed_bytes")?,
            norm_bytes: u("norm_bytes")?,
            data_start: (8 + len) as u64,
        })
    }
}

impl ModelFiles {
    /// Open a quantized model file made by `layerstream quantize`.
    pub fn open_quantized(path: &Path) -> Result<ModelFiles, String> {
        let h = QuantHeader::read(path)?;
        let cfg = Config::from_json(&Json::parse(&h.config_text)?)?;
        if cfg.format == WeightFormat::Bf16 {
            return Err("this file does not describe quantized weights".into());
        }
        let layout = layer_layout_for(&cfg);
        if layout.total != h.layer_bytes || cfg.vocab * cfg.row_bytes() != h.embed_bytes || cfg.hidden * 2 != h.norm_bytes {
            return Err("quantized header does not match the model's shape".into());
        }
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file_len = file.metadata().map_err(|e| e.to_string())?.len();
        let embed_start = h.data_start + (cfg.layers * layout.total) as u64;
        let norm_start = embed_start + h.embed_bytes as u64;
        if file_len != norm_start + h.norm_bytes as u64 {
            return Err(format!("file is {file_len} bytes but its header implies {}", norm_start + h.norm_bytes as u64));
        }
        let mut raw = vec![0u8; h.norm_bytes];
        std::os::unix::fs::FileExt::read_exact_at(&file, &mut raw, norm_start).map_err(|e| e.to_string())?;
        let mut final_norm = vec![0.0f32; cfg.hidden];
        crate::kernels::bf16_slice_to_f32(&raw, &mut final_norm);
        Ok(ModelFiles {
            layer_start: (0..cfg.layers).map(|l| h.data_start + (l * layout.total) as u64).collect(),
            embed_start,
            embed_len: h.embed_bytes,
            norm_start,
            norm_len: h.norm_bytes,
            final_norm,
            path: path.to_path_buf(),
            layout,
            cfg,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_formats_are_validated_and_sized() {
        let q8 = WeightFormat::Quant { bits: 8, group: 64 };
        let q4 = WeightFormat::Quant { bits: 4, group: 32 };
        assert_eq!(WeightFormat::Bf16.row_bytes(896), 1792);
        assert_eq!(q8.row_bytes(896), 896 + 14 * 2);
        assert_eq!(q4.row_bytes(896), 448 + 28 * 2);
        assert!(q8.validate(&[896, 4864]).is_ok() && q4.validate(&[896, 4864]).is_ok());
        for bad in [
            WeightFormat::Quant { bits: 6, group: 32 },
            WeightFormat::Quant { bits: 8, group: 24 },
            WeightFormat::Quant { bits: 8, group: 512 },
            WeightFormat::Quant { bits: 8, group: 256 }, // 896 is not a multiple of 256
        ] {
            assert!(bad.validate(&[896, 4864]).is_err(), "{bad:?}");
        }
    }

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
