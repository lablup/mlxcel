//! RoPE tables, byte-for-byte with HF `_compute_llama3_parameters` and the JAX
//! reference `llama3_inv_freq` / `rope_tables` in spike/openxla/model_jax.py.
//!
//! All math is done in f64 (matching numpy's default), then cos/sin are cast to
//! f32 for the baked constant tables, exactly as the JAX path did
//! (`jnp.asarray(np.cos(emb), jnp.float32)`). Keeping the trig in f64-then-f32
//! rather than emitting in-graph `stablehlo.cosine` removes any dependence on the
//! runtime's transcendental precision.

use super::config::Config;

/// inv_freq[i], i in 0..head_dim/2, with llama3 scaling. Returns f64.
pub fn llama3_inv_freq(c: &Config) -> Vec<f64> {
    let half = c.head_dim / 2;
    let mut inv = vec![0.0f64; half];
    let low_wl = c.orig_ctx as f64 / c.low_freq_factor;
    let high_wl = c.orig_ctx as f64 / c.high_freq_factor;
    for (i, slot) in inv.iter_mut().enumerate() {
        // base = 1 / theta^((2i)/head_dim)
        let exponent = (2 * i) as f64 / c.head_dim as f64;
        let base = 1.0 / c.rope_theta.powf(exponent);
        let wavelen = 2.0 * std::f64::consts::PI / base;

        // inv = where(wavelen > low_wl, base/factor, base)
        let mut v = if wavelen > low_wl {
            base / c.factor
        } else {
            base
        };
        // smoothed = (1 - smooth) * inv/factor + smooth * inv
        let smooth = (c.orig_ctx as f64 / wavelen - c.low_freq_factor)
            / (c.high_freq_factor - c.low_freq_factor);
        let smoothed = (1.0 - smooth) * v / c.factor + smooth * v;
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
    let inv = llama3_inv_freq(c);
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
