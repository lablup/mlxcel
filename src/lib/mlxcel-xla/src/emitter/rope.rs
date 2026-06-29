//! RoPE tables. The llama3 path is byte-for-byte with HF
//! `_compute_llama3_parameters` and the JAX reference `llama3_inv_freq` /
//! `rope_tables` in spike/openxla/model_jax.py; the plain path (Qwen2, #449 M3
//! Stage B) is the textbook `1 / theta^(2i/d)`.
//!
//! All math is done in f64 (matching numpy's default), then cos/sin are cast to
//! f32 for the baked constant tables, exactly as the JAX path did
//! (`jnp.asarray(np.cos(emb), jnp.float32)`). Keeping the trig in f64-then-f32
//! rather than emitting in-graph `stablehlo.cosine` removes any dependence on the
//! runtime's transcendental precision.

use super::config::{Config, RopeScaling};

/// inv_freq[i], i in 0..head_dim/2, in f64. Dispatches on the config's RoPE
/// scheme: plain RoPE (Qwen2) or llama3 scaling (Llama-3.x).
pub fn inv_freq(c: &Config) -> Vec<f64> {
    match &c.rope {
        RopeScaling::Plain => plain_inv_freq(c),
        RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            orig_ctx,
        } => llama3_inv_freq(c, *factor, *low_freq_factor, *high_freq_factor, *orig_ctx),
    }
}

/// Plain RoPE base frequencies: `inv_freq[i] = 1 / theta^((2i)/head_dim)`.
fn plain_inv_freq(c: &Config) -> Vec<f64> {
    let half = c.head_dim / 2;
    (0..half)
        .map(|i| {
            let exponent = (2 * i) as f64 / c.head_dim as f64;
            1.0 / c.rope_theta.powf(exponent)
        })
        .collect()
}

/// llama3-scaled base frequencies. Returns f64.
fn llama3_inv_freq(
    c: &Config,
    factor: f64,
    low_freq_factor: f64,
    high_freq_factor: f64,
    orig_ctx: usize,
) -> Vec<f64> {
    let half = c.head_dim / 2;
    let mut inv = vec![0.0f64; half];
    let low_wl = orig_ctx as f64 / low_freq_factor;
    let high_wl = orig_ctx as f64 / high_freq_factor;
    for (i, slot) in inv.iter_mut().enumerate() {
        // base = 1 / theta^((2i)/head_dim)
        let exponent = (2 * i) as f64 / c.head_dim as f64;
        let base = 1.0 / c.rope_theta.powf(exponent);
        let wavelen = 2.0 * std::f64::consts::PI / base;

        // inv = where(wavelen > low_wl, base/factor, base)
        let mut v = if wavelen > low_wl {
            base / factor
        } else {
            base
        };
        // smoothed = (1 - smooth) * inv/factor + smooth * inv
        let smooth =
            (orig_ctx as f64 / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
        let smoothed = (1.0 - smooth) * v / factor + smooth * v;
        // is_medium = high_wl <= wavelen <= low_wl (wavelen is finite, so this is
        // the same as the JAX `!(wavelen < high_wl) && !(wavelen > low_wl)`).
        let is_medium = wavelen >= high_wl && wavelen <= low_wl;
        if is_medium {
            v = smoothed;
        }
        *slot = v;
    }
    inv
}

/// Build cos and sin tables of shape [max_seq, head_dim] as flat row-major f32.
/// emb = concat([freqs, freqs], -1) where freqs = outer(pos, inv_freq).
pub fn rope_tables(c: &Config, max_seq: usize) -> (Vec<f32>, Vec<f32>) {
    let inv = inv_freq(c);
    let half = c.head_dim / 2;
    let d = c.head_dim;
    let mut cos = vec![0.0f32; max_seq * d];
    let mut sin = vec![0.0f32; max_seq * d];
    for p in 0..max_seq {
        for i in 0..half {
            let angle = p as f64 * inv[i];
            let (c_val, s_val) = (angle.cos() as f32, angle.sin() as f32);
            // first half [0, half) and second half [half, d) are identical
            cos[p * d + i] = c_val;
            cos[p * d + half + i] = c_val;
            sin[p * d + i] = s_val;
            sin[p * d + half + i] = s_val;
        }
    }
    (cos, sin)
}
