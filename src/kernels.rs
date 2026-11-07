//! Numeric kernels. Weights stay in bf16 (as stored on disk) and are widened to f32 on the
//! fly, so a layer takes 2 bytes per parameter in memory instead of 4.

use crate::pool::{Pool, SyncPtr};
use crate::weights::WeightFormat;

#[inline(always)]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline(always)]
fn bf16_at(bytes: &[u8], i: usize) -> f32 {
    bf16_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
}

/// Widen a bf16 byte slice into f32 values.
pub fn bf16_slice_to_f32(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(bytes.len(), out.len() * 2);
    for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        *o = bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
    }
}

/// Dot product of a bf16 row (as little-endian bytes) with an f32 vector.
#[inline]
pub fn dot_bf16(w: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len() * 2);
    let mut acc = [0.0f32; 8];
    let mut wc = w.chunks_exact(16);
    let mut xc = x.chunks_exact(8);
    for (wb, xb) in (&mut wc).zip(&mut xc) {
        for k in 0..8 {
            let u = u16::from_le_bytes([wb[2 * k], wb[2 * k + 1]]);
            acc[k] += bf16_to_f32(u) * xb[k];
        }
    }
    let mut sum = acc.iter().sum::<f32>();
    for (wb, xv) in wc.remainder().chunks_exact(2).zip(xc.remainder()) {
        sum += bf16_to_f32(u16::from_le_bytes([wb[0], wb[1]])) * xv;
    }
    sum
}

#[inline(always)]
fn scale_at(scales: &[u8], g: usize) -> f32 {
    bf16_to_f32(u16::from_le_bytes([scales[2 * g], scales[2 * g + 1]]))
}

/// Dot product of an 8-bit quantized row with an f32 vector. A stored row is `cols` signed
/// bytes followed by one bf16 scale per `group` of them.
#[inline]
pub fn dot_q8(row: &[u8], x: &[f32], group: usize) -> f32 {
    let cols = x.len();
    debug_assert_eq!(row.len(), cols + cols / group * 2);
    let (data, scales) = row.split_at(cols);
    let mut total = 0.0f32;
    for (g, (qg, xg)) in data.chunks_exact(group).zip(x.chunks_exact(group)).enumerate() {
        let mut acc = [0.0f32; 8];
        for (qc, xc) in qg.chunks_exact(8).zip(xg.chunks_exact(8)) {
            for k in 0..8 {
                acc[k] += (qc[k] as i8) as f32 * xc[k];
            }
        }
        total += scale_at(scales, g) * acc.iter().sum::<f32>();
    }
    total
}

/// Dot product of a 4-bit quantized row with an f32 vector. Inside a group of `group`
/// weights, byte `j` holds weight `j` in its low nibble and weight `j + group / 2` in its
/// high nibble, each stored as `value + 8`.
#[inline]
pub fn dot_q4(row: &[u8], x: &[f32], group: usize) -> f32 {
    let cols = x.len();
    debug_assert_eq!(row.len(), cols / 2 + cols / group * 2);
    let (data, scales) = row.split_at(cols / 2);
    let half = group / 2;
    let mut total = 0.0f32;
    for (g, (bytes, xg)) in data.chunks_exact(half).zip(x.chunks_exact(group)).enumerate() {
        let (x_lo, x_hi) = xg.split_at(half);
        let mut acc = [0.0f32; 8];
        for ((bc, lo), hi) in bytes.chunks_exact(8).zip(x_lo.chunks_exact(8)).zip(x_hi.chunks_exact(8)) {
            for k in 0..8 {
                let b = bc[k];
                acc[k] += ((b & 15) as i32 - 8) as f32 * lo[k] + ((b >> 4) as i32 - 8) as f32 * hi[k];
            }
        }
        total += scale_at(scales, g) * acc.iter().sum::<f32>();
    }
    total
}

/// Widen one stored row (of whichever format) into f32 values.
pub fn dequant_row(fmt: WeightFormat, raw: &[u8], out: &mut [f32]) {
    let WeightFormat::Quant { bits, group } = fmt else {
        return bf16_slice_to_f32(raw, out);
    };
    let cols = out.len();
    assert!(cols.is_multiple_of(group), "row width is not a multiple of the group size");
    assert_eq!(raw.len(), fmt.row_bytes(cols), "quantized row has the wrong size");
    let (data, scales) = raw.split_at(cols * bits as usize / 8);
    for (g, og) in out.chunks_exact_mut(group).enumerate() {
        let s = scale_at(scales, g);
        match bits {
            8 => {
                for (o, &q) in og.iter_mut().zip(&data[g * group..(g + 1) * group]) {
                    *o = (q as i8) as f32 * s;
                }
            }
            4 => {
                let half = group / 2;
                for (j, &b) in data[g * half..(g + 1) * half].iter().enumerate() {
                    og[j] = ((b & 15) as i32 - 8) as f32 * s;
                    og[j + half] = ((b >> 4) as i32 - 8) as f32 * s;
                }
            }
            _ => unreachable!("the bit width is validated when the config is read"),
        }
    }
}

const ROW_BLOCK: usize = 16;

/// `y[t][r] = dot(stored row r of w, x[t])` for `t` input vectors, where `dot` knows how one
/// stored row of `row_bytes` bytes is laid out. Work is split into blocks of output rows
/// across the pool.
#[allow(clippy::too_many_arguments)]
fn matmul_rows<F>(pool: &Pool, w: &[u8], rows: usize, row_bytes: usize, cols: usize, x: &[f32], t: usize, y: &mut [f32], dot: F)
where
    F: Fn(&[u8], &[f32]) -> f32 + Sync,
{
    assert_eq!(w.len(), rows * row_bytes, "weight matrix has the wrong size");
    assert_eq!(x.len(), t * cols);
    assert_eq!(y.len(), t * rows);
    let n_tasks = rows.div_ceil(ROW_BLOCK);

    if t == 1 {
        let out = SyncPtr::new(y);
        pool.run(n_tasks, &|task| {
            let r0 = task * ROW_BLOCK;
            let r1 = (r0 + ROW_BLOCK).min(rows);
            // SAFETY: tasks write disjoint row ranges of `y`.
            let dst = unsafe { std::slice::from_raw_parts_mut(out.get().add(r0), r1 - r0) };
            for (k, o) in dst.iter_mut().enumerate() {
                let r = r0 + k;
                *o = dot(&w[r * row_bytes..(r + 1) * row_bytes], x);
            }
        });
        return;
    }

    // Several inputs: compute as [rows][t] so each task owns a contiguous range, then
    // transpose into [t][rows].
    let mut tmp = vec![0.0f32; rows * t];
    let out = SyncPtr::new(&mut tmp);
    pool.run(n_tasks, &|task| {
        let r0 = task * ROW_BLOCK;
        let r1 = (r0 + ROW_BLOCK).min(rows);
        // SAFETY: tasks write disjoint row ranges of `tmp`.
        let dst = unsafe { std::slice::from_raw_parts_mut(out.get().add(r0 * t), (r1 - r0) * t) };
        for r in r0..r1 {
            let wr = &w[r * row_bytes..(r + 1) * row_bytes];
            for tt in 0..t {
                dst[(r - r0) * t + tt] = dot(wr, &x[tt * cols..(tt + 1) * cols]);
            }
        }
    });
    for tt in 0..t {
        for r in 0..rows {
            y[tt * rows + r] = tmp[r * t + tt];
        }
    }
}

/// `y[t][r] = sum_c w[r][c] * x[t][c]` for `t` input vectors, with `w` a `rows x cols`
/// bf16 matrix.
pub fn matmul_bf16(pool: &Pool, w: &[u8], rows: usize, cols: usize, x: &[f32], t: usize, y: &mut [f32]) {
    matmul_rows(pool, w, rows, cols * 2, cols, x, t, y, dot_bf16);
}

/// The same product with an 8-bit quantized matrix.
#[allow(clippy::too_many_arguments)]
pub fn matmul_q8(pool: &Pool, w: &[u8], rows: usize, cols: usize, group: usize, x: &[f32], t: usize, y: &mut [f32]) {
    matmul_rows(pool, w, rows, cols + cols / group * 2, cols, x, t, y, |r, xv| dot_q8(r, xv, group));
}

/// The same product with a 4-bit quantized matrix.
#[allow(clippy::too_many_arguments)]
pub fn matmul_q4(pool: &Pool, w: &[u8], rows: usize, cols: usize, group: usize, x: &[f32], t: usize, y: &mut [f32]) {
    matmul_rows(pool, w, rows, cols / 2 + cols / group * 2, cols, x, t, y, |r, xv| dot_q4(r, xv, group));
}

/// Matrix product with the weights in whichever format they are stored in.
#[allow(clippy::too_many_arguments)]
pub fn matmul(pool: &Pool, fmt: WeightFormat, w: &[u8], rows: usize, cols: usize, x: &[f32], t: usize, y: &mut [f32]) {
    match fmt {
        WeightFormat::Bf16 => matmul_bf16(pool, w, rows, cols, x, t, y),
        WeightFormat::Quant { bits: 8, group } => matmul_q8(pool, w, rows, cols, group, x, t, y),
        WeightFormat::Quant { bits: 4, group } => matmul_q4(pool, w, rows, cols, group, x, t, y),
        WeightFormat::Quant { bits, .. } => panic!("unsupported bit width {bits}"),
    }
}

/// Add a bf16 bias vector to each of `t` rows of `y`.
pub fn add_bias_bf16(y: &mut [f32], bias: &[u8], t: usize) {
    let n = bias.len() / 2;
    assert_eq!(y.len(), t * n);
    for row in y.chunks_exact_mut(n) {
        for (i, v) in row.iter_mut().enumerate() {
            *v += bf16_at(bias, i);
        }
    }
}

/// RMSNorm of each of the `t` rows of `x` into `out`, scaled by a bf16 weight vector.
pub fn rms_norm_bf16(x: &[f32], weight: &[u8], eps: f32, out: &mut [f32]) {
    let n = weight.len() / 2;
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % n, 0);
    for (xr, or) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let mean_sq = xr.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for (i, (o, v)) in or.iter_mut().zip(xr).enumerate() {
            *o = v * inv * bf16_at(weight, i);
        }
    }
}

/// RMSNorm with an f32 weight vector (for the small final norm that stays resident).
pub fn rms_norm_f32(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = weight.len();
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for ((o, v), w) in out.iter_mut().zip(x).zip(weight) {
        *o = v * inv * w;
    }
}

/// Rotary position embedding on one head vector, "rotate half" convention: element `i`
/// pairs with element `i + d/2`.
pub fn rope_inplace(head: &mut [f32], inv_freq: &[f32], pos: usize) {
    let half = head.len() / 2;
    debug_assert_eq!(inv_freq.len(), half);
    for i in 0..half {
        let angle = pos as f32 * inv_freq[i];
        let (s, c) = angle.sin_cos();
        let (a, b) = (head[i], head[i + half]);
        head[i] = a * c - b * s;
        head[i + half] = b * c + a * s;
    }
}

pub fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = 1.0 / sum;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect()
    }

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    #[test]
    fn dot_matches_a_double_precision_reference() {
        let mut seed = 7;
        for n in [1usize, 7, 8, 9, 16, 17, 64, 896, 4864] {
            let w: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
            let x: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
            let wb = to_bf16_bytes(&w);
            let want: f64 = (0..n).map(|i| bf16_at(&wb, i) as f64 * x[i] as f64).sum();
            let got = dot_bf16(&wb, &x) as f64;
            assert!((got - want).abs() < 1e-4 * (1.0 + want.abs()), "n={n}: {got} vs {want}");
        }
    }

    #[test]
    fn matmul_single_and_batched_agree_and_match_naive() {
        let pool = Pool::new(4);
        let (rows, cols) = (50, 40); // not multiples of the block size
        let mut seed = 99;
        let w: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut seed)).collect();
        let wb = to_bf16_bytes(&w);
        for t in [1usize, 2, 5] {
            let x: Vec<f32> = (0..t * cols).map(|_| lcg(&mut seed)).collect();
            let mut y = vec![0.0; t * rows];
            matmul_bf16(&pool, &wb, rows, cols, &x, t, &mut y);
            for tt in 0..t {
                for r in 0..rows {
                    let want: f64 = (0..cols).map(|c| bf16_at(&wb, r * cols + c) as f64 * x[tt * cols + c] as f64).sum();
                    assert!((y[tt * rows + r] as f64 - want).abs() < 1e-4, "t={t} row={r}");
                }
            }
        }
    }

    #[test]
    fn rms_norm_has_unit_rms_for_unit_weights() {
        let x = [3.0f32, -4.0, 12.0, 0.5];
        let ones = to_bf16_bytes(&[1.0; 4]);
        let mut out = [0.0; 4];
        rms_norm_bf16(&x, &ones, 0.0, &mut out);
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rope_at_position_zero_is_identity_and_preserves_norm() {
        let inv: Vec<f32> = (0..4).map(|i| 10000f32.powf(-(i as f32) / 4.0)).collect();
        let mut v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = v;
        rope_inplace(&mut v, &inv, 0);
        assert_eq!(v, orig);
        rope_inplace(&mut v, &inv, 37);
        let n0: f32 = orig.iter().map(|x| x * x).sum();
        let n1: f32 = v.iter().map(|x| x * x).sum();
        assert!((n0 - n1).abs() < 1e-3);
    }

    #[test]
    fn softmax_sums_to_one_and_is_stable() {
        let mut v = [1000.0f32, 1001.0, 999.0];
        softmax_inplace(&mut v);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(v[1] > v[0] && v[0] > v[2]);
    }
}
