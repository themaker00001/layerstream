//! Quantization: shrink a bf16 model by storing every weight as an 8-bit or 4-bit integer plus
//! one shared bf16 scale per group of weights.
//!
//! Every stored row (a row of a projection matrix, or one row of the embedding table) is
//! self-contained: its packed integers, then one bf16 scale per group.
//!
//! * 8-bit: one signed byte per weight, in -127..=127.
//! * 4-bit: inside a group, byte `j` holds weight `j` in its low nibble and weight
//!   `j + group / 2` in its high nibble, each stored as `value + 8` with value in -7..=7.
//!
//! The scale of a group is `clip * max|w| / qmax`, rounded to bf16 *before* the weights are
//! rounded to integers, so what is stored is exactly what is later multiplied back. With
//! clip search on, each group tries a few clipping ratios and keeps the one with the least
//! squared error: cutting off a rare outlier can buy finer steps for every other weight.

use crate::json::Json;
use crate::kernels::bf16_slice_to_f32;
use crate::pool::{Pool, SyncPtr};
use crate::source::{FileSource, Source};
use crate::weights::{layer_layout_for, matrix_shape, Config, QuantHeader, WeightFormat, N_TENSORS};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

/// Round an f32 to the nearest bf16 (ties to even) and widen it back.
pub fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return v; // NaN
    }
    let lsb = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
}

const CLIPS: [f32; 9] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6];

/// Quantize one group into `q` (values in -qmax..=qmax); returns its bf16-exact scale and the
/// squared reconstruction error.
fn quantize_group(bits: u8, w: &[f32], clip_search: bool, q: &mut [i8]) -> (f32, f64) {
    let qmax = if bits == 8 { 127.0f32 } else { 7.0 };
    let amax = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let zeros = |q: &mut [i8]| -> (f32, f64) {
        q.fill(0);
        (0.0, w.iter().map(|v| (v * v) as f64).sum::<f64>())
    };
    if amax == 0.0 || !amax.is_finite() {
        return zeros(q);
    }
    let clips: &[f32] = if clip_search { &CLIPS } else { &CLIPS[..1] };
    let mut best = (f64::INFINITY, 0.0f32);
    for &c in clips {
        let s = bf16_round(c * amax / qmax);
        if s == 0.0 {
            continue;
        }
        let inv = 1.0 / s;
        let mut err = 0.0f64;
        for &v in w {
            let d = v - (v * inv).round().clamp(-qmax, qmax) * s;
            err += (d * d) as f64;
        }
        if err < best.0 {
            best = (err, s);
        }
    }
    let s = best.1;
    if s == 0.0 {
        return zeros(q);
    }
    let inv = 1.0 / s;
    for (qv, &v) in q.iter_mut().zip(w) {
        *qv = (v * inv).round().clamp(-qmax, qmax) as i8;
    }
    (s, best.0)
}

/// Quantize one row of f32 weights into `out` (exactly `fmt.row_bytes(row.len())` bytes).
/// Returns `(squared error, squared norm)` of the row, for reporting.
pub fn quantize_row(fmt: WeightFormat, row: &[f32], clip_search: bool, out: &mut [u8]) -> (f64, f64) {
    let WeightFormat::Quant { bits, group } = fmt else { panic!("quantize_row needs a quantized format") };
    let cols = row.len();
    assert!(group <= 256 && cols.is_multiple_of(group), "row width must be a multiple of the group size");
    assert_eq!(out.len(), fmt.row_bytes(cols), "output row has the wrong size");
    let (data, scales) = out.split_at_mut(cols * bits as usize / 8);
    let mut q = [0i8; 256];
    let (mut err, mut norm) = (0.0f64, 0.0f64);
    for (g, wg) in row.chunks_exact(group).enumerate() {
        let (s, e) = quantize_group(bits, wg, clip_search, &mut q[..group]);
        err += e;
        norm += wg.iter().map(|v| (v * v) as f64).sum::<f64>();
        scales[2 * g..2 * g + 2].copy_from_slice(&((s.to_bits() >> 16) as u16).to_le_bytes());
        if bits == 8 {
            for (d, &v) in data[g * group..(g + 1) * group].iter_mut().zip(&q[..group]) {
                *d = v as u8;
            }
        } else {
            let half = group / 2;
            for j in 0..half {
                data[g * half + j] = ((q[j] + 8) as u8) | (((q[j + half] + 8) as u8) << 4);
            }
        }
    }
    (err, norm)
}

/// Quantize a `rows x cols` bf16 matrix into `dst`, rows in parallel. Returns the summed
/// `(squared error, squared norm)`.
#[allow(clippy::too_many_arguments)]
fn quantize_matrix(pool: &Pool, src: &[u8], rows: usize, cols: usize, fmt: WeightFormat, clip_search: bool, dst: &mut [u8]) -> (f64, f64) {
    let (rb_in, rb_out) = (cols * 2, fmt.row_bytes(cols));
    assert_eq!(src.len(), rows * rb_in);
    assert_eq!(dst.len(), rows * rb_out);
    const BLOCK: usize = 8;
    let out = SyncPtr::new(dst);
    let acc = Mutex::new((0.0f64, 0.0f64));
    pool.run(rows.div_ceil(BLOCK), &|task| {
        let (r0, r1) = (task * BLOCK, (task * BLOCK + BLOCK).min(rows));
        let mut f = vec![0.0f32; cols];
        let (mut e, mut n) = (0.0, 0.0);
        for r in r0..r1 {
            bf16_slice_to_f32(&src[r * rb_in..(r + 1) * rb_in], &mut f);
            // SAFETY: tasks write disjoint rows of `dst`.
            let d = unsafe { std::slice::from_raw_parts_mut(out.get().add(r * rb_out), rb_out) };
            let (a, b) = quantize_row(fmt, &f, clip_search, d);
            e += a;
            n += b;
        }
        let mut g = acc.lock().unwrap_or_else(|p| p.into_inner());
        g.0 += e;
        g.1 += n;
    });
    acc.into_inner().unwrap_or_else(|p| p.into_inner())
}

/// Add the weight format to a `config.json` text, as the extra key `layerstream_quant`. With
/// `embed` the embedding table gets its own `(bits, group)`; without it, it shares the layers'.
pub fn inject_quant(config_text: &str, bits: u8, group: usize, embed: Option<(u8, usize)>) -> Result<String, String> {
    let end = config_text.rfind('}').ok_or("config.json has no closing brace")?;
    let head = config_text[..end].trim_end();
    let extra = match embed {
        Some((eb, eg)) => format!(",\"embed_bits\":{eb},\"embed_group\":{eg}"),
        None => String::new(),
    };
    Ok(format!("{head},\"layerstream_quant\":{{\"bits\":{bits},\"group\":{group}{extra}}}}}"))
}

#[derive(Debug, Clone, Copy)]
pub struct QuantSummary {
    pub in_bytes: u64,
    pub out_bytes: u64,
    /// Root-mean-square weight error relative to the weights' own RMS, over every matrix and
    /// the embedding table.
    pub rel_rmse: f64,
}

/// Quantize the bf16 model behind `src` into a new file at `out` (it must not exist).
#[allow(clippy::too_many_arguments)]
pub fn quantize_model(
    src: &FileSource,
    config_text: &str,
    fmt: WeightFormat,
    embed_fmt: WeightFormat,
    clip_search: bool,
    threads: usize,
    out: &Path,
) -> Result<QuantSummary, String> {
    let WeightFormat::Quant { bits, group } = fmt else { return Err("quantize needs a quantized target format".into()) };
    let WeightFormat::Quant { bits: eb, group: eg } = embed_fmt else { return Err("the embedding table needs a quantized format too".into()) };
    let files = src.files();
    if files.cfg.format != WeightFormat::Bf16 {
        return Err("the source model is already quantized".into());
    }
    let qtext = inject_quant(config_text, bits, group, if embed_fmt == fmt { None } else { Some((eb, eg)) })?;
    let qcfg = Config::from_json(&Json::parse(&qtext)?)?;
    let mut as_bf16 = qcfg.clone();
    as_bf16.format = WeightFormat::Bf16;
    as_bf16.embed_format = WeightFormat::Bf16;
    if as_bf16 != files.cfg {
        return Err("the config text does not describe the model being quantized".into());
    }
    let src_layout = &files.layout;
    let dst_layout = layer_layout_for(&qcfg);

    let header = QuantHeader {
        config_text: qtext,
        layer_bytes: dst_layout.total,
        embed_bytes: qcfg.vocab * qcfg.row_bytes(),
        norm_bytes: qcfg.hidden * 2,
        data_start: 0,
    };
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out)
        .map_err(|e| format!("{}: {e}", out.display()))?;
    let head_bytes = header.to_bytes();
    file.write_all(&head_bytes).map_err(|e| e.to_string())?;
    let mut written = head_bytes.len() as u64;

    let pool = Pool::new(threads.max(1));
    let (mut err, mut norm) = (0.0f64, 0.0f64);
    let mut inb = vec![0u8; src_layout.total];
    let mut outb = vec![0u8; dst_layout.total];
    for l in 0..qcfg.layers {
        src.read_layer(l, &mut inb)?;
        for slot in 0..N_TENSORS {
            let (si, sl) = src_layout.ranges[slot];
            let (di, dl) = dst_layout.ranges[slot];
            match matrix_shape(slot, &qcfg) {
                None => {
                    if sl != dl {
                        return Err(format!("tensor {slot} changed size"));
                    }
                    outb[di..di + dl].copy_from_slice(&inb[si..si + sl]);
                }
                Some((rows, cols)) => {
                    let (e, n) = quantize_matrix(&pool, &inb[si..si + sl], rows, cols, fmt, clip_search, &mut outb[di..di + dl]);
                    err += e;
                    norm += n;
                }
            }
        }
        file.write_all(&outb).map_err(|e| e.to_string())?;
        written += outb.len() as u64;
    }

    let (rb_in, rb_out) = (files.cfg.row_bytes(), qcfg.row_bytes());
    let rows_per = 4096.min(qcfg.vocab);
    let mut ein = vec![0u8; rows_per * rb_in];
    let mut eout = vec![0u8; rows_per * rb_out];
    let mut first = 0;
    while first < qcfg.vocab {
        let n = rows_per.min(qcfg.vocab - first);
        src.read_embed_rows(first, n, &mut ein[..n * rb_in])?;
        let (e, m) = quantize_matrix(&pool, &ein[..n * rb_in], n, qcfg.hidden, embed_fmt, clip_search, &mut eout[..n * rb_out]);
        err += e;
        norm += m;
        file.write_all(&eout[..n * rb_out]).map_err(|e| e.to_string())?;
        written += (n * rb_out) as u64;
        first += n;
    }

    let mut final_norm = vec![0u8; files.norm_len];
    src.read_final_norm_raw(&mut final_norm)?;
    file.write_all(&final_norm).map_err(|e| e.to_string())?;
    written += final_norm.len() as u64;
    file.sync_all().map_err(|e| e.to_string())?;

    let in_bytes = (qcfg.layers * src_layout.total + files.embed_len + files.norm_len) as u64;
    Ok(QuantSummary { in_bytes, out_bytes: written, rel_rmse: if norm > 0.0 { (err / norm).sqrt() } else { 0.0 } })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{dequant_row, matmul};

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    fn row(n: usize, seed: &mut u64) -> Vec<f32> {
        // a few large outliers among small weights, like real weights
        (0..n).map(|i| lcg(seed) * if i % 37 == 0 { 0.5 } else { 0.05 }).collect()
    }

    fn fmt(bits: u8, group: usize) -> WeightFormat {
        WeightFormat::Quant { bits, group }
    }

    #[test]
    fn bf16_rounding_is_round_to_nearest_even() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(1.0 + 2f32.powi(-8)), 1.0); // halfway, ties to the even neighbour
        assert_eq!(bf16_round(1.0 + 3.0 * 2f32.powi(-8)), 1.0 + 2f32.powi(-6));
        assert_eq!(bf16_round(1.0 + 2f32.powi(-9)), 1.0);
        assert!(bf16_round(f32::NAN).is_nan());
        assert_eq!(bf16_round(f32::INFINITY), f32::INFINITY);
        let v = 0.123_456_79_f32;
        assert_eq!(bf16_round(bf16_round(v)), bf16_round(v));
    }

    #[test]
    fn round_trip_error_is_bounded_by_half_a_step() {
        let mut seed = 5;
        for (bits, group) in [(8u8, 16usize), (8, 32), (8, 64), (4, 16), (4, 32), (4, 64)] {
            let f = fmt(bits, group);
            let w = row(256, &mut seed);
            let mut packed = vec![0u8; f.row_bytes(256)];
            quantize_row(f, &w, false, &mut packed);
            let mut back = vec![0.0f32; 256];
            dequant_row(f, &packed, &mut back);
            for (g, (wg, bg)) in w.chunks(group).zip(back.chunks(group)).enumerate() {
                let amax = wg.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let step = amax / if bits == 8 { 127.0 } else { 7.0 };
                for (a, b) in wg.iter().zip(bg) {
                    // half a step, plus the tiny error of rounding the scale itself to bf16
                    assert!((a - b).abs() <= step * 0.52, "bits={bits} group={group} g={g}: {a} vs {b}");
                }
                // the largest weight of a group is reproduced to within the scale's bf16 rounding
                let big = wg.iter().cloned().fold(0.0f32, |m, v| if v.abs() > m.abs() { v } else { m });
                let back_big = wg.iter().zip(bg).find(|(a, _)| **a == big).unwrap().1;
                assert!((big - back_big).abs() <= amax * 0.01, "group max not preserved: {big} vs {back_big}");
            }
        }
    }

    #[test]
    fn clip_search_never_does_worse_and_helps_four_bit() {
        let mut seed = 11;
        let (mut plain, mut clipped) = (0.0, 0.0);
        for _ in 0..40 {
            let w = row(128, &mut seed);
            let f = fmt(4, 32);
            let mut a = vec![0u8; f.row_bytes(128)];
            let mut b = a.clone();
            let (ea, _) = quantize_row(f, &w, false, &mut a);
            let (eb, _) = quantize_row(f, &w, true, &mut b);
            assert!(eb <= ea * (1.0 + 1e-9), "clip search made a row worse");
            plain += ea;
            clipped += eb;
        }
        assert!(clipped < plain * 0.97, "clip search should reduce 4-bit error on outlier-heavy rows: {clipped} vs {plain}");
    }

    #[test]
    fn zero_and_tiny_rows_do_not_produce_nan() {
        for f in [fmt(8, 32), fmt(4, 32)] {
            for w in [vec![0.0f32; 64], vec![1e-45f32; 64], vec![-3.0f32; 64]] {
                let mut packed = vec![0u8; f.row_bytes(64)];
                quantize_row(f, &w, true, &mut packed);
                let mut back = vec![0.0f32; 64];
                dequant_row(f, &packed, &mut back);
                assert!(back.iter().all(|v| v.is_finite()), "{f:?}");
            }
        }
        let mut packed = vec![0u8; fmt(8, 32).row_bytes(32)];
        quantize_row(fmt(8, 32), &[0.0; 32], true, &mut packed);
        let mut back = [1.0f32; 32];
        dequant_row(fmt(8, 32), &packed, &mut back);
        assert!(back.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn quantized_matmul_equals_a_matmul_over_the_dequantized_weights() {
        let pool = Pool::new(4);
        let mut seed = 21;
        for (bits, group) in [(8u8, 32usize), (8, 64), (4, 32), (4, 64)] {
            let f = fmt(bits, group);
            let (rows, cols) = (37, 128); // rows not a multiple of the block size
            let rb = f.row_bytes(cols);
            let mut packed = vec![0u8; rows * rb];
            let mut deq = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let w = row(cols, &mut seed);
                quantize_row(f, &w, true, &mut packed[r * rb..(r + 1) * rb]);
                dequant_row(f, &packed[r * rb..(r + 1) * rb], &mut deq[r * cols..(r + 1) * cols]);
            }
            for t in [1usize, 3] {
                let x: Vec<f32> = (0..t * cols).map(|_| lcg(&mut seed)).collect();
                let mut y = vec![0.0f32; t * rows];
                matmul(&pool, f, &packed, rows, cols, &x, t, &mut y);
                for tt in 0..t {
                    for r in 0..rows {
                        let want: f64 = (0..cols).map(|c| deq[r * cols + c] as f64 * x[tt * cols + c] as f64).sum();
                        let got = y[tt * rows + r] as f64;
                        assert!((got - want).abs() < 1e-4 * (1.0 + want.abs()), "bits={bits} t={t} row={r}: {got} vs {want}");
                    }
                }
            }
        }
    }

    #[test]
    fn inject_quant_produces_a_config_that_parses_back() {
        let base = r#"{"model_type":"qwen2","hidden_act":"silu","tie_word_embeddings":true,"hidden_size":896,
            "num_attention_heads":14,"num_key_value_heads":2,"intermediate_size":4864,"num_hidden_layers":24,
            "vocab_size":151936,"rms_norm_eps":1e-6,"rope_theta":1000000.0,"max_position_embeddings":32768}
"#;
        let text = inject_quant(base, 4, 32, None).unwrap();
        let c = Config::from_json(&Json::parse(&text).unwrap()).unwrap();
        assert_eq!((c.format, c.embed_format), (fmt(4, 32), fmt(4, 32)));
        assert_eq!(c.row_bytes(), 448 + 56);
        // the embedding table can have its own format
        let text = inject_quant(base, 4, 32, Some((8, 64))).unwrap();
        let c = Config::from_json(&Json::parse(&text).unwrap()).unwrap();
        assert_eq!((c.format, c.embed_format), (fmt(4, 32), fmt(8, 64)));
        assert_eq!(c.row_bytes(), 896 + 28);
        let plain = Config::from_json(&Json::parse(base).unwrap()).unwrap();
        assert_eq!((plain.format, plain.embed_format), (WeightFormat::Bf16, WeightFormat::Bf16));
        // a group that does not divide the layer widths is refused
        assert!(Config::from_json(&Json::parse(&inject_quant(base, 8, 256, None).unwrap()).unwrap()).is_err());
        // embed_bits without embed_group is refused
        let half = text.replace(",\"embed_group\":64", "");
        assert!(Config::from_json(&Json::parse(&half).unwrap()).is_err());
    }
}
