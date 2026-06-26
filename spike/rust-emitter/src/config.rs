//! Llama-3.2-1B-Instruct config (matches spike/openxla/model_jax.py Config).

#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub inter: usize,
    pub n_layers: usize,
    pub n_q: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub vocab: usize,
    // llama3 rope scaling
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub orig_ctx: usize,
}

impl Config {
    /// Hard-coded Llama-3.2-1B-Instruct values (config.json of the spike model).
    pub fn llama_3_2_1b() -> Self {
        Config {
            hidden: 2048,
            inter: 8192,
            n_layers: 16,
            n_q: 32,
            n_kv: 8,
            head_dim: 64,
            eps: 1e-5,
            rope_theta: 500000.0,
            vocab: 128256,
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            orig_ctx: 8192,
        }
    }

    pub fn group(&self) -> usize {
        self.n_q / self.n_kv
    }

    /// Attention scale head_dim^-0.5.
    pub fn scale(&self) -> f32 {
        (self.head_dim as f32).powf(-0.5)
    }
}
