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
//! weights as f32 (widening bf16 / f16, copying f32, or dequantizing MLX 4 / 8-bit
//! affine weights) in the emitter's arg order, from either a single-file
//! `model.safetensors` or a sharded checkpoint (via its
//! `model.safetensors.index.json`, which is how the big untied models ship), and
//! (4) hands all of it to the
//! shim, which keeps the weights resident on the device and threads the KV cache
//! across steps. Then [`IreeLlama::prefill`] / [`IreeLlama::decode`] are token-in
//! / token-out. Emitting from config (issue #449 M3 Stage 2d) replaced the bundled
//! Llama-3.2-1B `.mlir` assets, so any checkpoint of a supported dense family loads:
//! Llama, Qwen2, Qwen3, Gemma1/2/3, SmolLM3, OLMo2/3, Seed-OSS, MiMo, InternLM3,
//! ExaOne (issues #497 / #499), and the parallel-block / norm-variant pack Cohere,
//! Cohere2, Phi3, StableLM, StarCoder2, Granite, MiniCPM (issue #498). Each family's
//! per-layer weight order in `weight_specs` (`weights.rs`) mirrors the emitter's arg
//! schedule: the q/k/v biases, the Qwen3 / Gemma3 / OLMo2/3 q/k norms, the Gemma2/3
//! feed-forward norms, the OLMo2/3 absence of an `input_layernorm`, the #498
//! LayerNorm / o_proj / MLP biases and dense StarCoder2 MLP, and the fused Phi3
//! `qkv_proj` / `gate_up_proj` (read once and row-sliced into the separate args). An
//! untied checkpoint (`tie_word_embeddings = false`, e.g. Llama-3.1-8B, larger
//! Qwen2.5, OLMo2/3, Phi3, StableLM, MiniCPM) adds its `lm_head.weight`, matching the
//! separate `params['lm_head']` arg the emitter takes for the final projection.
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

use crate::emitter::{
    Config, emit_decode_ragged_with, emit_decode_with, emit_prefill_with, resolve_precision,
};
// The loader reads the per-architecture checkpoint-weight order from
// `weights::weight_specs`, which sources its names from `weight_names::scheme_names`
// (issue #499 naming schemes) and covers the #498 dense pack (LayerNorm / o_proj /
// MLP biases, the dense StarCoder2 MLP, and the row-sliced fused Phi3 projections)
// and the #500 MoE expert bank (the stacked `switch_mlp` weights, dequantized with
// `dequantize_affine_stacked`). Both are pure-Rust and unit-tested without `iree`.
use crate::weights::{
    WeightSpec, bf16_to_f32, dequantize_affine, dequantize_affine_stacked, f16_to_f32,
    f32_le_to_f32, slice_rows, weight_specs,
};

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

// The per-architecture checkpoint-weight order the loader reads lives in
// `weights::weight_specs` (pure logic, unit-tested without the IREE runtime, and
// kept in lock-step with the emitter's arg order). It covers the dense arch pack
// (issue #498): LayerNorm biases, the o_proj / MLP biases, the dense StarCoder2
// MLP, and the Phi3 fused `qkv_proj` / `gate_up_proj` (loaded once and row-sliced
// into the emitter's separate args); and the MoE expert bank (issue #500): the
// router and the stacked `switch_mlp` gate/up/down (plus the optional shared
// expert), appended after each MoE layer's attention weights / norms.

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
/// cuda codegen); `metal` -> the Metal target (metal-spirv codegen; the Apple
/// Silicon dev path, where the macOS runtime registers the metal driver and the
/// pinned macOS universal2 iree-compile has metal-spirv codegen).
fn target_flags(device: &str) -> Result<&'static [&'static str], String> {
    if device == "cuda" {
        Ok(&["--iree-hal-target-device=cuda"])
    } else if device == "metal" {
        Ok(&["--iree-hal-target-device=metal"])
    } else if device.starts_with("local") {
        Ok(&[
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
        ])
    } else {
        Err(format!(
            "unsupported OpenXLA device {device:?}; use \"local-task\" (CPU), \
             \"metal\" (Apple Silicon GPU, macOS build), or \"cuda\" (cuda build)"
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
    let precision = resolve_precision(device);
    let prefill = emit_prefill_with(cfg, true, precision);
    let decode = emit_decode_with(cfg, true, precision);
    compile_pair(device, &prefill, "prefill", &decode, "decode")
}

/// Map each needed weight name to the safetensors file that holds it, in the same
/// order as `names`. A single-file checkpoint (`model.safetensors` present) maps
/// every name to that one file; a sharded checkpoint (no `model.safetensors`, only
/// `model-0000k-of-...safetensors` shards) reads `model.safetensors.index.json`'s
/// `weight_map`. The big untied models (e.g. Llama-3.1-8B) ship sharded, so this
/// is what lets them load. A model dir that has BOTH a `model.safetensors` and an
/// index uses the single file (the index then just points back at it).
fn resolve_weight_shards(model_dir: &Path, names: &[String]) -> Result<Vec<PathBuf>, String> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single; names.len()]);
    }
    let index = model_dir.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index).map_err(|e| {
        format!(
            "no model.safetensors and no readable model.safetensors.index.json in {}: {e}",
            model_dir.display()
        )
    })?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", index.display()))?;
    let map = v
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("{}: missing object `weight_map`", index.display()))?;
    names
        .iter()
        .map(|name| {
            map.get(name)
                .and_then(serde_json::Value::as_str)
                .map(|f| model_dir.join(f))
                .ok_or_else(|| format!("{}: weight_map has no entry for `{name}`", index.display()))
        })
        .collect()
}

/// Load the weights bf16 -> f32 in the emitter's arg order, returning the owned
/// f32 buffers (kept alive until the shim copies them) plus the flat (ptr, rank,
/// dims) arrays the C ABI takes. `cfg` fixes the layer count and weight order.
///
/// Single-file and sharded checkpoints both load: [`resolve_weight_shards`] maps
/// each weight to its file, and the weights are read shard by shard (each shard
/// mmap'd exactly once, its tensors copied out as owned f32) and placed back into
/// the emitter's arg order by index, so the buffers line up with the graph args
/// regardless of how the checkpoint was split.
#[allow(clippy::type_complexity)]
fn load_weights(
    model_dir: &Path,
    cfg: &Config,
) -> Result<(Vec<Vec<f32>>, Vec<c_int>, Vec<i64>), String> {
    let specs = weight_specs(cfg);
    let names: Vec<String> = specs.iter().map(|s| s.tensor_name().to_string()).collect();
    let shard_paths = resolve_weight_shards(model_dir, &names)?;

    // Group weight indices by shard so each shard file is opened/mmap'd once; the
    // results are placed by index, preserving the emitter's arg order. A Phi3
    // fused checkpoint references one tensor (`qkv_proj` / `gate_up_proj`) from
    // several specs; each reads and row-slices the (memmapped) shard.
    let mut by_shard: std::collections::BTreeMap<&Path, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, p) in shard_paths.iter().enumerate() {
        by_shard.entry(p.as_path()).or_default().push(i);
    }

    let n = specs.len();
    let mut bufs: Vec<Vec<f32>> = vec![Vec::new(); n];
    let mut ranks: Vec<c_int> = vec![0; n];
    let mut dims: Vec<i64> = vec![0; n * 4];
    for (shard, idxs) in by_shard {
        let file = File::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
        // Safety: the file is read-only for the lifetime of the mmap below.
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {e}", shard.display()))?;
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| format!("parse {}: {e}", shard.display()))?;
        for &i in &idxs {
            let name = &names[i];
            let t = st
                .tensor(name)
                .map_err(|e| format!("weight {name} in {}: {e}", shard.display()))?;

            // Widen the whole tensor to row-major `[out, in]` (or `[out]`) f32 (the
            // graph's dtype): an MLX affine `U32`-packed weight is dequantized (the
            // layernorms / biases are not quantized and take the widen path).
            let (data, shape): (Vec<f32>, Vec<usize>) = if t.dtype() == Dtype::U32 {
                let qc = cfg.quantization.ok_or_else(|| {
                    format!(
                        "weight {name} is U32 (quantized) but config.json has no `quantization`"
                    )
                })?;
                let prefix = name
                    .strip_suffix(".weight")
                    .ok_or_else(|| format!("quantized weight {name} does not end in `.weight`"))?;
                let scales_name = format!("{prefix}.scales");
                let biases_name = format!("{prefix}.biases");
                let scales = st.tensor(&scales_name).map_err(|e| {
                    format!("{scales_name} (for {name}) in {}: {e}", shard.display())
                })?;
                let biases = st.tensor(&biases_name).map_err(|e| {
                    format!("{biases_name} (for {name}) in {}: {e}", shard.display())
                })?;
                // mlx-lm emits the affine scales/biases as either f16 or bf16
                // (Qwen3 / Gemma3-27B / Qwen3-MoE checkpoints use bf16); accept a
                // matching 16-bit pair and widen it to f32 in the dequantizer.
                let sb_bf16 = match (scales.dtype(), biases.dtype()) {
                    (Dtype::F16, Dtype::F16) => false,
                    (Dtype::BF16, Dtype::BF16) => true,
                    (s, b) => {
                        return Err(format!(
                            "{prefix} scales/biases dtype {s:?}/{b:?}, expected a matching F16 or BF16 pair"
                        ));
                    }
                };
                let shape = t.shape();
                match shape.len() {
                    // Ordinary `[out, in_packed]` weight (attention, dense MLP,
                    // shared expert, router when quantized).
                    2 => {
                        let (out, in_packed) = (shape[0], shape[1]);
                        let in_ = in_packed * (32 / qc.bits);
                        let d = dequantize_affine(
                            t.data(),
                            scales.data(),
                            biases.data(),
                            out,
                            in_packed,
                            qc.bits,
                            qc.group_size,
                            sb_bf16,
                        )
                        .map_err(|e| format!("dequantize {name}: {e}"))?;
                        (d, vec![out, in_])
                    }
                    // Stacked `[experts, out, in_packed]` MoE `switch_mlp` weight
                    // (issue #500); dequantize each expert slab and keep the leading
                    // expert axis so it feeds the emitter's `[E, out, in]` arg.
                    3 => {
                        let (experts, out, in_packed) = (shape[0], shape[1], shape[2]);
                        let in_ = in_packed * (32 / qc.bits);
                        let d = dequantize_affine_stacked(
                            t.data(),
                            scales.data(),
                            biases.data(),
                            experts,
                            out,
                            in_packed,
                            qc.bits,
                            qc.group_size,
                            sb_bf16,
                        )
                        .map_err(|e| format!("dequantize stacked {name}: {e}"))?;
                        (d, vec![experts, out, in_])
                    }
                    r => {
                        return Err(format!("quantized weight {name} rank {r} not in {{2, 3}}"));
                    }
                }
            } else {
                // bf16 and f16 are the common checkpoint dtypes; f32 is a
                // passthrough. The widening is exact for all three, matching HF's
                // f32 reference.
                let d = match t.dtype() {
                    Dtype::BF16 => bf16_to_f32(t.data()),
                    Dtype::F16 => f16_to_f32(t.data()),
                    Dtype::F32 => f32_le_to_f32(t.data()),
                    other => {
                        return Err(format!(
                            "weight {name} dtype {other:?}, expected BF16/F16/F32 or \
                             MLX-quantized U32"
                        ));
                    }
                };
                let shape = t.shape();
                if shape.len() > 4 {
                    return Err(format!("weight {name} rank {} > 4", shape.len()));
                }
                (d, shape.to_vec())
            };

            // Place the whole tensor, or (Phi3 fused) a row-slice, into arg slot i.
            match &specs[i] {
                WeightSpec::Whole(_) => {
                    ranks[i] = shape.len() as c_int;
                    for (k, &s) in shape.iter().enumerate() {
                        dims[i * 4 + k] = s as i64;
                    }
                    bufs[i] = data;
                }
                WeightSpec::Rows { start, end, .. } => {
                    if shape.len() != 2 {
                        return Err(format!(
                            "row-slice weight {name} is rank {} (expected 2)",
                            shape.len()
                        ));
                    }
                    bufs[i] = slice_rows(&data, shape[0], *start, *end)
                        .map_err(|e| format!("row-slice {name}: {e}"))?;
                    ranks[i] = 2;
                    dims[i * 4] = (*end - *start) as i64;
                    dims[i * 4 + 1] = shape[1] as i64;
                }
            }
        }
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
        // Emit the logits prefill + the ragged decode graph for this model + b_max
        // at the device-resolved precision (f16 on GPU by default, f32 on CPU).
        let precision = resolve_precision(device);
        let prefill_mlir = emit_prefill_with(&cfg, false, precision);
        let decode_mlir = emit_decode_ragged_with(&cfg, b_max, false, precision);
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
