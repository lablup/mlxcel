// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! IREE execution path (issue #449 Phase 3 M2), compiled only under the `iree`
//! feature.
//!
//! [`IreeLlama`] is the safe Rust owner of the C shim (`csrc/xla_iree.c`). On
//! [`IreeLlama::load`] it (1) reads the model's `config.json` into an emitter
//! [`Config`], (2) emits the `prefill` / `decode_step` StableHLO graphs from that
//! config (the #451 emitter, ending in an on-device argmax) and compiles them to
//! vmfbs with the dist's `iree-compile`, cached by content hash, (3) loads the
//! bf16 weights as f32 in the emitter's arg order, and (4) hands all of it to the
//! shim, which keeps the weights resident on the device and threads the KV cache
//! across steps. Then [`IreeLlama::prefill`] / [`IreeLlama::decode`] are token-in
//! / token-out. Emitting from config (issue #449 M3 Stage 2d) replaced the bundled
//! Llama-3.2-1B `.mlir` assets, so any checkpoint of a supported architecture
//! loads: Llama (any size) and Qwen2 (plain RoPE + q/k/v bias; Stage B), the
//! latter adding its bias tensors to `weight_names` to match the emitted graph.
//!
//! Proven token-exact against the HF temp-0 reference in
//! `spike/openxla/artifacts/results.json` before being vendored from the
//! standalone gate (`spike/iree-ffi`).

use std::ffi::CString;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::os::raw::{c_char, c_int};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::emitter::{Config, emit_decode, emit_decode_ragged, emit_prefill};

/// Prefill bucket baked into the emitted `prefill` graph (`tensor<256xi32>`,
/// == MAX_SEQ, so it covers any prompt the 256-slot KV cache holds).
pub const PREFILL_LP: usize = 256;

/// Opaque handle to the C-side execution context.
#[repr(C)]
struct XlaCtx {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn xla_llama_create(
        device: *const c_char,
        prefill: *const c_char,
        decode: *const c_char,
        n_weights: c_int,
        weight_data: *const *const f32,
        weight_ranks: *const c_int,
        weight_dims: *const i64,
    ) -> *mut XlaCtx;
    fn xla_llama_prefill(
        c: *mut XlaCtx,
        tokens: *const c_int,
        lp: c_int,
        positions: *const c_int,
        real_len: c_int,
        out_token: *mut c_int,
    ) -> c_int;
    fn xla_llama_decode(
        c: *mut XlaCtx,
        token: c_int,
        pos: c_int,
        cache_len: c_int,
        out_token: *mut c_int,
    ) -> c_int;
    // Ragged continuous-batching (#449 M3 Stage 2b/2d). `ragged_reset` sizes the
    // batch; the `_logits` calls run one slot's prefill / all B slots' decode and
    // copy the per-row `[vocab]` logits to host so the engine samples there (Stage
    // 2d). `prefill_slot_logits` also does the device-side KV slot write.
    fn xla_llama_ragged_reset(c: *mut XlaCtx, bsz: c_int) -> c_int;
    fn xla_llama_prefill_slot_logits(
        c: *mut XlaCtx,
        slot: c_int,
        tokens: *const c_int,
        lp: c_int,
        positions: *const c_int,
        real_len: c_int,
        vocab: c_int,
        out_logits: *mut f32,
    ) -> c_int;
    fn xla_llama_decode_ragged_logits(
        c: *mut XlaCtx,
        bsz: c_int,
        tokens: *const c_int,
        pos: *const c_int,
        cache_len: *const c_int,
        vocab: c_int,
        out_logits: *mut f32,
    ) -> c_int;
    fn xla_llama_free(c: *mut XlaCtx);
}

/// The slot counts (`B_max`) the serve worker maps a request's batch size to. The
/// ragged decode graph for the chosen `b_max` is emitted from the model config at
/// load (any `b_max` is emittable; the worker selects from this set).
pub(crate) const RAGGED_B_VALUES: &[usize] = &[4, 8];

/// The weight names in the emitter's exact arg order: embed, final_norm, then per
/// layer down, gate, in_ln, post_ln, up, wk, wo, wq, wv, and — for a `qkv_bias`
/// architecture (Qwen2) — the k/q/v projection biases. The layer count and the
/// presence of biases come from the model config so the order matches the emitted
/// graph's args (`take_layer_weights` in `emitter/model.rs`).
fn weight_names(cfg: &Config) -> Vec<String> {
    let mut names = vec![
        "model.embed_tokens.weight".to_string(),
        "model.norm.weight".to_string(),
    ];
    for i in 0..cfg.n_layers {
        let p = format!("model.layers.{i}.");
        for suf in [
            "mlp.down_proj.weight",
            "mlp.gate_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.up_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ] {
            names.push(format!("{p}{suf}"));
        }
        // Qwen2 q/k/v projection biases, appended per layer in the same k/q/v
        // order `take_layer_weights` adds them to the emitted graph args.
        if cfg.qkv_bias {
            for suf in [
                "self_attn.k_proj.bias",
                "self_attn.q_proj.bias",
                "self_attn.v_proj.bias",
            ] {
                names.push(format!("{p}{suf}"));
            }
        }
    }
    names
}

/// bf16 little-endian bytes -> f32 (bf16 is the high 16 bits of f32).
fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// Locate the IREE distribution: a runtime `IREE_DIST` override first, else the
/// path baked at build time (the dist whose runtime is linked into this binary).
/// Only the dist (CPU/vulkan) build uses this; the cuda build links a source-built
/// runtime and takes its iree-compile from `MLXCEL_XLA_IREE_COMPILE` instead.
#[cfg(not(xla_iree_cuda))]
fn iree_dist() -> Result<PathBuf, String> {
    if let Ok(d) = std::env::var("IREE_DIST") {
        return Ok(PathBuf::from(d));
    }
    match option_env!("MLXCEL_XLA_IREE_DIST") {
        Some(d) => Ok(PathBuf::from(d)),
        None => Err("IREE_DIST is not set and no dist path was baked at build time".to_string()),
    }
}

/// The `iree-compile` binary used to lower the bundled graphs to vmfbs. In the
/// cuda build the source-built runtime ships no compiler, so a cuda-capable
/// iree-compile (version-matched to that runtime) is required via
/// `MLXCEL_XLA_IREE_COMPILE` (runtime env, else baked at build); in the dist
/// build it is the dist's own `bin/iree-compile`.
fn iree_compile_bin() -> Result<PathBuf, String> {
    if let Ok(ic) = std::env::var("MLXCEL_XLA_IREE_COMPILE") {
        return Ok(PathBuf::from(ic));
    }
    if let Some(ic) = option_env!("MLXCEL_XLA_IREE_COMPILE") {
        return Ok(PathBuf::from(ic));
    }
    #[cfg(xla_iree_cuda)]
    {
        Err("the cuda build needs a cuda-capable iree-compile: set \
             MLXCEL_XLA_IREE_COMPILE to one matching the source runtime version \
             (e.g. the pip iree-compile)"
            .to_string())
    }
    #[cfg(not(xla_iree_cuda))]
    {
        let ic = iree_dist()?.join("bin/iree-compile");
        if !ic.exists() {
            return Err(format!(
                "iree-compile not found at {} (set IREE_DIST to a valid dist)",
                ic.display()
            ));
        }
        Ok(ic)
    }
}

/// iree-compile target flags for a HAL device. `local-task`/`local-sync` -> the
/// CPU (llvm-cpu) target; `cuda` -> the CUDA target (only usable in a cuda
/// build, whose runtime registers the cuda driver and whose iree-compile has
/// cuda codegen).
fn target_flags(device: &str) -> Result<&'static [&'static str], String> {
    if device == "cuda" {
        Ok(&["--iree-hal-target-device=cuda"])
    } else if device.starts_with("local") {
        Ok(&[
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
        ])
    } else {
        Err(format!(
            "unsupported OpenXLA device {device:?}; use \"local-task\" (CPU) or, in a \
             cuda build, \"cuda\""
        ))
    }
}

/// Compile one bundled graph to a vmfb, cached by a hash of its text + flags so
/// repeated loads skip the ~3 s compile.
fn compile_one(
    iree_compile: &Path,
    mlir: &str,
    flags: &[&str],
    cache: &Path,
    tag: &str,
) -> Result<PathBuf, String> {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    mlir.hash(&mut h);
    for f in flags {
        f.hash(&mut h);
    }
    // Include the compiler path so a cuda vmfb is never reused for a cpu build
    // (or across iree-compile versions) just because the graph text matches.
    iree_compile.to_string_lossy().hash(&mut h);
    let key = h.finish();
    let mlir_path = cache.join(format!("{tag}-{key:016x}.mlir"));
    let vmfb_path = cache.join(format!("{tag}-{key:016x}.vmfb"));
    if vmfb_path.exists() {
        return Ok(vmfb_path);
    }
    std::fs::write(&mlir_path, mlir).map_err(|e| format!("write {}: {e}", mlir_path.display()))?;
    let out = Command::new(iree_compile)
        .arg("--iree-input-type=stablehlo")
        .args(flags)
        .arg(&mlir_path)
        .arg("-o")
        .arg(&vmfb_path)
        .output()
        .map_err(|e| format!("run {}: {e}", iree_compile.display()))?;
    if !out.status.success() {
        return Err(format!(
            "iree-compile failed for {tag}: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(vmfb_path)
}

/// Compile a (prefill, decode) graph pair for `device`. The single-sequence engine
/// uses the argmax pair; the ragged engine uses the logits pair (Stage 2d).
fn compile_pair(
    device: &str,
    prefill_mlir: &str,
    prefill_tag: &str,
    decode_mlir: &str,
    decode_tag: &str,
) -> Result<(PathBuf, PathBuf), String> {
    let iree_compile = iree_compile_bin()?;
    if !iree_compile.exists() {
        return Err(format!(
            "iree-compile not found at {}",
            iree_compile.display()
        ));
    }
    let flags = target_flags(device)?;
    let cache = std::env::temp_dir().join("mlxcel-xla-vmfb");
    std::fs::create_dir_all(&cache).map_err(|e| format!("mkdir {}: {e}", cache.display()))?;
    let pre = compile_one(&iree_compile, prefill_mlir, flags, &cache, prefill_tag)?;
    let dec = compile_one(&iree_compile, decode_mlir, flags, &cache, decode_tag)?;
    Ok((pre, dec))
}

/// Emit and compile the argmax prefill + single-token decode graphs for `cfg` on
/// `device` (the single-sequence engine's on-device-argmax pair). The graph text
/// is emitted from the model config, so `compile_one`'s text-hash cache keys a
/// distinct vmfb per architecture.
fn compile_vmfbs(device: &str, cfg: &Config) -> Result<(PathBuf, PathBuf), String> {
    let prefill = emit_prefill(cfg, true);
    let decode = emit_decode(cfg, true);
    compile_pair(device, &prefill, "prefill", &decode, "decode")
}

/// Load the weights bf16 -> f32 in the emitter's arg order, returning the owned
/// f32 buffers (kept alive until the shim copies them) plus the flat (ptr, rank,
/// dims) arrays the C ABI takes. `cfg` fixes the layer count and weight order.
#[allow(clippy::type_complexity)]
fn load_weights(
    model_dir: &Path,
    cfg: &Config,
) -> Result<(Vec<Vec<f32>>, Vec<c_int>, Vec<i64>), String> {
    let st_path = model_dir.join("model.safetensors");
    let file = File::open(&st_path).map_err(|e| format!("open {}: {e}", st_path.display()))?;
    // Safety: the file is read-only for the lifetime of the mmap below.
    let mmap =
        unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {e}", st_path.display()))?;
    let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse safetensors: {e}"))?;

    let names = weight_names(cfg);
    let mut bufs: Vec<Vec<f32>> = Vec::with_capacity(names.len());
    let mut ranks: Vec<c_int> = Vec::with_capacity(names.len());
    let mut dims: Vec<i64> = Vec::with_capacity(names.len() * 4);
    for name in &names {
        let t = st.tensor(name).map_err(|e| format!("weight {name}: {e}"))?;
        if t.dtype() != Dtype::BF16 {
            return Err(format!(
                "weight {name} dtype {:?}, expected BF16",
                t.dtype()
            ));
        }
        let shape = t.shape();
        if shape.len() > 4 {
            return Err(format!("weight {name} rank {} > 4", shape.len()));
        }
        ranks.push(shape.len() as c_int);
        let mut d4 = [0i64; 4];
        for (k, &s) in shape.iter().enumerate() {
            d4[k] = s as i64;
        }
        dims.extend_from_slice(&d4);
        bufs.push(bf16_to_f32(t.data()));
    }
    Ok((bufs, ranks, dims))
}

fn path_cstring(p: &Path) -> Result<CString, String> {
    CString::new(p.as_os_str().as_bytes())
        .map_err(|_| format!("path has an interior nul byte: {}", p.display()))
}

/// Load the weights and create the C execution context for a (prefill, decode)
/// vmfb pair on `device`. Shared by the single-sequence ([`IreeLlama`]) and ragged
/// ([`IreeRaggedLlama`]) engines, which differ only in which decode vmfb they pass.
fn create_ctx(
    model_dir: &Path,
    cfg: &Config,
    device: &str,
    prefill_vmfb: &Path,
    decode_vmfb: &Path,
) -> Result<*mut XlaCtx, String> {
    let (bufs, ranks, dims) = load_weights(model_dir, cfg)?;
    let ptrs: Vec<*const f32> = bufs.iter().map(|b| b.as_ptr()).collect();
    let c_dev = CString::new(device).map_err(|_| "device has interior nul".to_string())?;
    let c_pre = path_cstring(prefill_vmfb)?;
    let c_dec = path_cstring(decode_vmfb)?;
    // Safety: pointers are valid for the duration of the call; the shim copies
    // the weight data into device buffers before returning.
    let ctx = unsafe {
        xla_llama_create(
            c_dev.as_ptr(),
            c_pre.as_ptr(),
            c_dec.as_ptr(),
            bufs.len() as c_int,
            ptrs.as_ptr(),
            ranks.as_ptr(),
            dims.as_ptr(),
        )
    };
    // Weights are resident on the device now; free the host copy.
    drop(ptrs);
    drop(bufs);
    if ctx.is_null() {
        return Err(
            "xla_llama_create failed (IREE runtime; see stderr for the status)".to_string(),
        );
    }
    Ok(ctx)
}

/// Owns the IREE execution context: one session with the prefill + decode
/// modules, the resident weights, and the threaded KV cache. Not `Send`/`Sync`
/// (the raw context is single-threaded), matching the single-sequence session.
pub struct IreeLlama {
    ctx: *mut XlaCtx,
}

impl IreeLlama {
    /// Prepare execution for a model directory on a HAL `device`
    /// (`"local-task"` for CPU). Compiles the bundled graphs, uploads the
    /// weights resident, and readies the prefill / decode calls.
    pub fn load(model_dir: &Path, device: &str) -> Result<Self, String> {
        let cfg = Config::from_json(model_dir)?;
        let (prefill_vmfb, decode_vmfb) = compile_vmfbs(device, &cfg)?;
        let ctx = create_ctx(model_dir, &cfg, device, &prefill_vmfb, &decode_vmfb)?;
        Ok(Self { ctx })
    }

    /// Seed the KV cache with `token_ids` (length <= [`PREFILL_LP`]) via the
    /// bucketed prefill graph. The returned token (the graph's argmax at
    /// `real_len-1`) is unused by the seed-then-decode drive loop.
    pub fn prefill_seed(&mut self, token_ids: &[i32]) -> Result<(), String> {
        self.prefill_padded(token_ids).map(|_| ())
    }

    /// Run the bucketed prefill over the FULL prompt and return its first token
    /// (the argmax at `prompt.len() - 1`). Unlike [`prefill_seed`], which seeds
    /// the KV and discards the token for the seed-then-decode loop, this returns
    /// the token, matching the batched engine's slot-seed convention
    /// ([`IreeRaggedLlama::prefill_slot`]); so a single-seq stream captured with
    /// `prefill_first` + [`decode`](Self::decode) is the right reference for it.
    ///
    /// [`prefill_seed`]: Self::prefill_seed
    pub fn prefill_first(&mut self, prompt: &[i32]) -> Result<i32, String> {
        if prompt.is_empty() {
            return Err("prefill_first requires a non-empty prompt".to_string());
        }
        self.prefill_padded(prompt)
    }

    /// Pad `prompt` into the [`PREFILL_LP`] bucket, run the prefill, and return its
    /// first token. Accepts an empty prompt (the seed-then-decode loop prefills a
    /// zero-length prefix when the prompt is a single token).
    fn prefill_padded(&mut self, prompt: &[i32]) -> Result<i32, String> {
        if prompt.len() > PREFILL_LP {
            return Err(format!(
                "the OpenXLA M2 prefill graph is bucketed at {PREFILL_LP} tokens; prompt \
                 prefix of {} exceeds it (a larger-bucket graph is a follow-up)",
                prompt.len()
            ));
        }
        let mut tokens = vec![0i32; PREFILL_LP];
        tokens[..prompt.len()].copy_from_slice(prompt);
        let positions: Vec<c_int> = (0..PREFILL_LP as c_int).collect();
        let mut out = 0i32;
        // Safety: buffers outlive the call; the shim stores the returned KV.
        let rc = unsafe {
            xla_llama_prefill(
                self.ctx,
                tokens.as_ptr(),
                PREFILL_LP as c_int,
                positions.as_ptr(),
                prompt.len() as c_int,
                &mut out,
            )
        };
        if rc != 0 {
            return Err(format!("xla_llama_prefill failed (status {rc})"));
        }
        Ok(out)
    }

    /// Advance one token at `cache_len` (== position), returning the next token
    /// id (on-device argmax) and writing the new K/V into the resident cache.
    pub fn decode(&mut self, token: i32, cache_len: i32) -> Result<i32, String> {
        let mut out = 0i32;
        // Safety: the shim threads its own resident KV; only scalars cross here.
        let rc = unsafe { xla_llama_decode(self.ctx, token, cache_len, cache_len, &mut out) };
        if rc != 0 {
            return Err(format!("xla_llama_decode failed (status {rc})"));
        }
        Ok(out)
    }
}

impl Drop for IreeLlama {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // Safety: ctx was produced by xla_llama_create and not freed yet.
            unsafe { xla_llama_free(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}

/// Owns a ragged (continuous-batching) IREE context (#449 M3 Stage 2b/2d): the
/// logits prefill module plus a fixed-`B_max` ragged decode module, the resident
/// weights, and the rank-5 per-slot KV. Slots are seeded with [`prefill_slot_logits`]
/// (a device-side KV write) and advanced together by [`decode_ragged_logits`], each
/// row at its own position; both return logits so the wrapping engine samples on the
/// host. Not `Send`/`Sync` (the raw context is single-threaded); the engine that
/// wraps it owns it on one thread.
///
/// [`prefill_slot_logits`]: Self::prefill_slot_logits
/// [`decode_ragged_logits`]: Self::decode_ragged_logits
pub struct IreeRaggedLlama {
    ctx: *mut XlaCtx,
    b_max: usize,
    /// Vocabulary size (logits per row), from the model config; the readback
    /// buffers and the per-row logits slice are sized by it.
    vocab: usize,
}

impl IreeRaggedLlama {
    /// Prepare a ragged engine for `model_dir` on `device` with `b_max` slots.
    ///
    /// Verifies the architecture, compiles the bundled prefill + the ragged decode
    /// graph for `b_max`, uploads the weights resident, and sizes the batch.
    /// `b_max` must be one of the bundled graphs ([`RAGGED_B_VALUES`]).
    pub fn load(model_dir: &Path, device: &str, b_max: usize) -> Result<Self, String> {
        let cfg = Config::from_json(model_dir)?;
        if !RAGGED_B_VALUES.contains(&b_max) {
            return Err(format!(
                "the OpenXLA serve worker selects B_max from {RAGGED_B_VALUES:?}; \
                 {b_max} is not one of them"
            ));
        }
        // Emit the logits prefill + the ragged decode graph for this model + b_max.
        let prefill_mlir = emit_prefill(&cfg, false);
        let decode_mlir = emit_decode_ragged(&cfg, b_max, false);
        let (prefill_vmfb, decode_vmfb) = compile_pair(
            device,
            &prefill_mlir,
            "prefill_logits",
            &decode_mlir,
            &format!("decode_ragged_logits_b{b_max}"),
        )?;
        let ctx = create_ctx(model_dir, &cfg, device, &prefill_vmfb, &decode_vmfb)?;
        // Safety: ctx is a fresh valid context from create_ctx; free it on error.
        let rc = unsafe { xla_llama_ragged_reset(ctx, b_max as c_int) };
        if rc != 0 {
            unsafe { xla_llama_free(ctx) };
            return Err(format!("xla_llama_ragged_reset failed (status {rc})"));
        }
        Ok(Self {
            ctx,
            b_max,
            vocab: cfg.vocab,
        })
    }

    /// The fixed slot count this engine was compiled for.
    #[must_use]
    pub fn b_max(&self) -> usize {
        self.b_max
    }

    /// The vocabulary size (logits per row). The engine slices the ragged decode's
    /// flat `[b_max * vocab]` logits by this.
    #[must_use]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Seed slot `slot` with `prompt` (1..=[`PREFILL_LP`] tokens) and return its
    /// first-token `[vocab]` LOGITS (#449 M3 Stage 2d). The prompt's KV is written
    /// device-side into the slot's region of the rank-5 cache; the other slots are
    /// untouched, so a mid-stream admit does not disturb live sequences. The caller
    /// samples the first token from the returned logits.
    pub fn prefill_slot_logits(&mut self, slot: usize, prompt: &[i32]) -> Result<Vec<f32>, String> {
        if slot >= self.b_max {
            return Err(format!("slot {slot} out of range [0,{})", self.b_max));
        }
        if prompt.is_empty() {
            return Err("prefill_slot_logits requires a non-empty prompt".to_string());
        }
        if prompt.len() > PREFILL_LP {
            return Err(format!(
                "prompt of {} exceeds the {PREFILL_LP}-token prefill bucket",
                prompt.len()
            ));
        }
        let mut tokens = vec![0i32; PREFILL_LP];
        tokens[..prompt.len()].copy_from_slice(prompt);
        let positions: Vec<c_int> = (0..PREFILL_LP as c_int).collect();
        let mut logits = vec![0f32; self.vocab];
        // Safety: input buffers outlive the call; `logits` has self.vocab elements,
        // which the shim fills; the shim also writes the slot's KV device-side.
        let rc = unsafe {
            xla_llama_prefill_slot_logits(
                self.ctx,
                slot as c_int,
                tokens.as_ptr(),
                PREFILL_LP as c_int,
                positions.as_ptr(),
                prompt.len() as c_int,
                self.vocab as c_int,
                logits.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!(
                "xla_llama_prefill_slot_logits failed (status {rc})"
            ));
        }
        Ok(logits)
    }

    /// Advance all `b_max` slots one token, returning the flat `[b_max * vocab]`
    /// LOGITS (#449 M3 Stage 2d). `tokens` / `pos` / `cache_len` are per-row (length
    /// `b_max`); an inactive slot carries zeros (a masked no-op whose logits the
    /// caller discards). The caller samples a token per row from `logits[s*vocab..]`.
    pub fn decode_ragged_logits(
        &mut self,
        tokens: &[i32],
        pos: &[i32],
        cache_len: &[i32],
    ) -> Result<Vec<f32>, String> {
        if tokens.len() != self.b_max || pos.len() != self.b_max || cache_len.len() != self.b_max {
            return Err(format!(
                "decode_ragged_logits expects per-row arrays of length b_max = {}",
                self.b_max
            ));
        }
        let mut logits = vec![0f32; self.b_max * self.vocab];
        // Safety: the three input slices are length b_max == bsz; `logits` has
        // b_max*self.vocab elements, which the shim fills; it threads its rank-5 KV.
        let rc = unsafe {
            xla_llama_decode_ragged_logits(
                self.ctx,
                self.b_max as c_int,
                tokens.as_ptr(),
                pos.as_ptr(),
                cache_len.as_ptr(),
                self.vocab as c_int,
                logits.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!(
                "xla_llama_decode_ragged_logits failed (status {rc})"
            ));
        }
        Ok(logits)
    }
}

impl Drop for IreeRaggedLlama {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // Safety: ctx was produced by create_ctx and not freed yet.
            unsafe { xla_llama_free(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}
