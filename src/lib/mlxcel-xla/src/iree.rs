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
//! [`IreeLlama::load`] it (1) checks the model's `config.json` matches the
//! architecture the bundled StableHLO graphs were authored for, (2) compiles the
//! bundled `prefill` / `decode_step` graphs (which end in an on-device argmax,
//! the #451 emitter output) to vmfbs with the dist's `iree-compile`, cached by
//! content hash, (3) loads the bf16 weights as f32 in the emitter's arg order,
//! and (4) hands all of it to the shim, which keeps the weights resident on the
//! device and threads the KV cache across steps. Then [`IreeLlama::prefill`] /
//! [`IreeLlama::decode`] are token-in / token-out.
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

/// Llama-3.2-1B-Instruct shape the bundled graphs are authored for.
const N_LAYERS: usize = 16;
/// Prefill bucket baked into the bundled `prefill` graph (`tensor<256xi32>`,
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
    fn xla_llama_free(c: *mut XlaCtx);
}

/// The #451-emitted graphs (on-device argmax variant), bundled as crate assets.
/// `iree-compile` turns these into vmfbs that match the linked runtime.
const PREFILL_MLIR: &str = include_str!("../assets/llama-3.2-1b/prefill.mlir");
const DECODE_MLIR: &str = include_str!("../assets/llama-3.2-1b/decode.mlir");

/// The 146 weight names in the emitter's exact arg order: embed, final_norm,
/// then per layer down, gate, in_ln, post_ln, up, wk, wo, wq, wv.
fn weight_names() -> Vec<String> {
    let mut names = vec![
        "model.embed_tokens.weight".to_string(),
        "model.norm.weight".to_string(),
    ];
    for i in 0..N_LAYERS {
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

/// Verify `config.json` matches the architecture the bundled graphs encode. The
/// graphs hard-code Llama-3.2-1B-Instruct dimensions; a different model would
/// silently produce wrong shapes, so this fails loudly instead.
fn verify_config(model_dir: &Path) -> Result<(), String> {
    let p = model_dir.join("config.json");
    let s = std::fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&s).map_err(|e| format!("parse {}: {e}", p.display()))?;
    let want = [
        ("num_hidden_layers", 16u64),
        ("hidden_size", 2048),
        ("intermediate_size", 8192),
        ("num_attention_heads", 32),
        ("num_key_value_heads", 8),
        ("head_dim", 64),
        ("vocab_size", 128256),
    ];
    for (k, w) in want {
        let got = v.get(k).and_then(serde_json::Value::as_u64);
        if got != Some(w) {
            return Err(format!(
                "the OpenXLA backend's bundled graphs are authored for \
                 Llama-3.2-1B-Instruct (issue #449 M2): config.json `{k}` = {got:?}, \
                 expected {w} ({})",
                p.display()
            ));
        }
    }
    if v.get("tie_word_embeddings")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        return Err(format!(
            "the OpenXLA backend's bundled Llama-3.2-1B graph assumes tied word \
             embeddings (config.json `tie_word_embeddings` != true, {})",
            p.display()
        ));
    }
    Ok(())
}

/// Locate the IREE distribution: a runtime `IREE_DIST` override first, else the
/// path baked at build time (the dist whose runtime is linked into this binary).
fn iree_dist() -> Result<PathBuf, String> {
    if let Ok(d) = std::env::var("IREE_DIST") {
        return Ok(PathBuf::from(d));
    }
    match option_env!("MLXCEL_XLA_IREE_DIST") {
        Some(d) => Ok(PathBuf::from(d)),
        None => Err("IREE_DIST is not set and no dist path was baked at build time".to_string()),
    }
}

/// iree-compile target flags for a HAL device. M2 ships the CPU path; the
/// prebuilt aarch64 dist registers only local (CPU) and vulkan drivers, not
/// CUDA, so a GPU device needs a CUDA/Vulkan-enabled runtime (a follow-up).
fn target_flags(device: &str) -> Result<&'static [&'static str], String> {
    if device.starts_with("local") {
        Ok(&[
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
        ])
    } else {
        Err(format!(
            "the OpenXLA backend M2 path runs on CPU (device \"local-task\"); device \
             {device:?} needs a CUDA/Vulkan-enabled IREE runtime, which the prebuilt \
             dist does not register"
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

/// Compile the bundled prefill + decode graphs for `device`.
fn compile_vmfbs(device: &str) -> Result<(PathBuf, PathBuf), String> {
    let dist = iree_dist()?;
    let iree_compile = dist.join("bin/iree-compile");
    if !iree_compile.exists() {
        return Err(format!(
            "iree-compile not found at {} (set IREE_DIST to a valid dist)",
            iree_compile.display()
        ));
    }
    let flags = target_flags(device)?;
    let cache = std::env::temp_dir().join("mlxcel-xla-vmfb");
    std::fs::create_dir_all(&cache).map_err(|e| format!("mkdir {}: {e}", cache.display()))?;
    let pre = compile_one(&iree_compile, PREFILL_MLIR, flags, &cache, "prefill")?;
    let dec = compile_one(&iree_compile, DECODE_MLIR, flags, &cache, "decode")?;
    Ok((pre, dec))
}

/// Load the 146 weights bf16 -> f32 in the emitter's arg order, returning the
/// owned f32 buffers (kept alive until the shim copies them) plus the flat
/// (ptr, rank, dims) arrays the C ABI takes.
#[allow(clippy::type_complexity)]
fn load_weights(model_dir: &Path) -> Result<(Vec<Vec<f32>>, Vec<c_int>, Vec<i64>), String> {
    let st_path = model_dir.join("model.safetensors");
    let file = File::open(&st_path).map_err(|e| format!("open {}: {e}", st_path.display()))?;
    // Safety: the file is read-only for the lifetime of the mmap below.
    let mmap =
        unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {e}", st_path.display()))?;
    let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse safetensors: {e}"))?;

    let names = weight_names();
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
        verify_config(model_dir)?;
        let (prefill_vmfb, decode_vmfb) = compile_vmfbs(device)?;
        let (bufs, ranks, dims) = load_weights(model_dir)?;
        let ptrs: Vec<*const f32> = bufs.iter().map(|b| b.as_ptr()).collect();

        let c_dev = CString::new(device).map_err(|_| "device has interior nul".to_string())?;
        let c_pre = path_cstring(&prefill_vmfb)?;
        let c_dec = path_cstring(&decode_vmfb)?;
        // Safety: pointers are valid for the duration of the call; the shim
        // copies the weight data into device buffers before returning.
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
        Ok(Self { ctx })
    }

    /// Seed the KV cache with `token_ids` (length <= [`PREFILL_LP`]) via the
    /// bucketed prefill graph. The returned token (the graph's argmax at
    /// `real_len-1`) is unused by the seed-then-decode drive loop.
    pub fn prefill_seed(&mut self, token_ids: &[i32]) -> Result<(), String> {
        if token_ids.len() > PREFILL_LP {
            return Err(format!(
                "the OpenXLA M2 prefill graph is bucketed at {PREFILL_LP} tokens; prompt \
                 prefix of {} exceeds it (a larger-bucket graph is a follow-up)",
                token_ids.len()
            ));
        }
        let mut tokens = vec![0i32; PREFILL_LP];
        tokens[..token_ids.len()].copy_from_slice(token_ids);
        let positions: Vec<c_int> = (0..PREFILL_LP as c_int).collect();
        let mut out = 0i32;
        // Safety: buffers outlive the call; the shim stores the returned KV.
        let rc = unsafe {
            xla_llama_prefill(
                self.ctx,
                tokens.as_ptr(),
                PREFILL_LP as c_int,
                positions.as_ptr(),
                token_ids.len() as c_int,
                &mut out,
            )
        };
        if rc != 0 {
            return Err(format!("xla_llama_prefill failed (status {rc})"));
        }
        Ok(())
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
