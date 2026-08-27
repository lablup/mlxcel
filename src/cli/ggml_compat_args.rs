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

//! llama-server b10621 GGML runtime, placement, and memory options
//! (issue #1445).
//!
//! Every option here describes something about the **GGML** backend: which CPU
//! cores its thread pool runs on, how many model layers to copy into VRAM, how
//! to split a model across several GPUs, whether to `mmap` or `mlock` the GGUF
//! file, which RPC servers to farm work out to, and which GGML quantizer to
//! store the KV cache in. mlxcel runs its tensor work through MLX on one Metal
//! or CUDA device, loads MLX SafeTensors, and has no GGML backend at all, so
//! almost none of it has a counterpart to translate to.
//!
//! # The rule
//!
//! Accepting an option and ignoring it is the failure mode this module exists
//! to remove: a deployment that passes `--n-gpu-layers 20` believes twelve
//! layers are on the CPU, and one that passes `--cache-type-k q8_0` believes
//! its KV cache is 8-bit. Before #1445, six of these parsed and did nothing
//! and the other thirty-seven were rejected by clap as unknown arguments, so
//! neither group told the operator anything true.
//!
//! Each option is now accepted and its **value** is classified:
//!
//! - A value whose b10621 meaning mlxcel already satisfies, or that asks for
//!   nothing (`--split-mode none` on a single device, `--threads -1`,
//!   `--cpu-strict 0`, `--flash-attn on`), is **inert** and is accepted
//!   silently.
//! - Any other value would change b10621's behavior in a way mlxcel cannot
//!   reproduce, so it is **rejected at startup**, before a weight is read,
//!   with a diagnostic naming the option, the value, the platform limitation,
//!   and the supported mlxcel alternative where one exists.
//!
//! Everything is `hide = true`: these are compatibility surfaces, not mlxcel
//! features, and rendering them in `--help` would imply a GGML backend that
//! does not exist. mlxcel's own Metal, Accelerate, CUDA, neural-accelerator,
//! and TurboQuant options are unaffected and stay visible.
//!
//! # Environment bindings
//!
//! Value-taking options bind their `LLAMA_ARG_*` variable through clap, which
//! hands the string over unchanged, so there is no parsing difference to
//! reconcile. Value-less flags and `--x` / `--no-x` pairs do not: b10621 fires
//! a value-less option from the environment only when the value is exactly
//! `on`, `enabled`, `true` or `1` (`common_arg_utils::is_truthy`), and reads a
//! bool pair through `parse_bool_value` plus a `LLAMA_ARG_NO_*` alias, while
//! clap's boolish parser accepts a wider vocabulary and *errors* outside it.
//! [`env_flag`] and [`env_bool_pair`] reproduce b10621's own rules instead.
//!
//! Used by: mlxcel serve, mlxcel-server.
//!
//! Upstream reference: <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>

use std::path::Path;

use clap::Args;

/// b10621's truthy set for a value-less option read from the environment.
/// Compared case-sensitively, exactly as `common_arg_utils::is_truthy` does.
const TRUTHY: [&str; 4] = ["on", "enabled", "true", "1"];

/// b10621's falsey set (`common_arg_utils::is_falsey`).
const FALSEY: [&str; 4] = ["off", "disabled", "false", "0"];

/// True when `var` is set to a value b10621 would treat as enabling a
/// value-less option.
///
/// Everything outside [`TRUTHY`], an empty value and `0` included, leaves the
/// flag alone, so a variable inherited as `LLAMA_ARG_CPU_MOE=0` does not turn
/// the option on.
#[must_use]
pub fn env_flag(var: &str) -> bool {
    std::env::var(var).is_ok_and(|value| TRUTHY.contains(&value.as_str()))
}

/// Resolve a b10621 `--x` / `--no-x` pair from the environment.
///
/// Returns `None` when neither variable is set, `Some(Ok(value))` for a
/// recognized value, and `Some(Err(raw))` for a value b10621's
/// `parse_bool_value` would throw on. The `LLAMA_ARG_NO_*` alias wins and
/// means `false`, matching `common_arg::get_value_from_env`.
#[must_use]
pub fn env_bool_pair(var: &str) -> Option<Result<bool, String>> {
    if std::env::var(var.replace("LLAMA_ARG_", "LLAMA_ARG_NO_")).is_ok() {
        return Some(Ok(false));
    }
    let raw = std::env::var(var).ok()?;
    if TRUTHY.contains(&raw.as_str()) {
        Some(Ok(true))
    } else if FALSEY.contains(&raw.as_str()) {
        Some(Ok(false))
    } else {
        Some(Err(raw))
    }
}

/// llama-server b10621 GGML runtime / placement / memory options.
///
/// Flattened into both server binaries so the two surfaces cannot drift; see
/// `tests/cli_help_consistency.rs`.
#[derive(Args, Debug, Clone, Default)]
pub struct GgmlCompatArgs {
    // ── sampling ────────────────────────────────────────────────────────
    /// b10621 `--backend-sampling`: run the sampler on the compute backend.
    /// Inert: mlxcel's sampler is always MLX ops on the accelerator.
    #[arg(long = "backend-sampling", hide = true)]
    pub backend_sampling: bool,

    // ── loading ─────────────────────────────────────────────────────────
    /// b10621 `--check-tensors`: scan tensor data for invalid values.
    #[arg(long = "check-tensors", hide = true)]
    pub check_tensors: bool,

    /// b10621 `--load-mode`: auto | none | mmap | mlock | mmap+mlock | dio.
    #[arg(
        long = "load-mode",
        env = "LLAMA_ARG_LOAD_MODE",
        value_name = "MODE",
        hide = true
    )]
    pub load_mode: Option<String>,

    /// b10621 `--mlock`: keep the model resident in RAM.
    #[arg(long = "mlock", hide = true)]
    pub mlock: bool,

    /// b10621 `--mmap` (positive half of the pair).
    #[arg(long = "mmap", overrides_with = "no_mmap", hide = true)]
    pub mmap: bool,

    /// b10621 `--no-mmap` (negative half of the pair).
    #[arg(long = "no-mmap", overrides_with = "mmap", hide = true)]
    pub no_mmap: bool,

    /// b10621 `--direct-io` (positive half of the pair).
    #[arg(long = "direct-io", overrides_with = "no_direct_io", hide = true)]
    pub direct_io: bool,

    /// b10621 `--no-direct-io` (negative half of the pair).
    #[arg(long = "no-direct-io", overrides_with = "direct_io", hide = true)]
    pub no_direct_io: bool,

    /// b10621 `--repack` (positive half of the pair).
    #[arg(long = "repack", overrides_with = "no_repack", hide = true)]
    pub repack: bool,

    /// b10621 `--no-repack` (negative half of the pair).
    #[arg(long = "no-repack", overrides_with = "repack", hide = true)]
    pub no_repack: bool,

    // ── attention kernel ────────────────────────────────────────────────
    /// b10621 `--flash-attn`: on | off | auto.
    #[arg(
        long = "flash-attn",
        env = "LLAMA_ARG_FLASH_ATTN",
        value_name = "[on|off|auto]",
        hide = true
    )]
    pub flash_attn: Option<String>,

    // ── device placement ────────────────────────────────────────────────
    /// b10621 `--device`: comma-separated offload device list.
    #[arg(
        long = "device",
        env = "LLAMA_ARG_DEVICE",
        value_name = "<dev1,dev2,..>",
        hide = true
    )]
    pub device: Option<String>,

    /// b10621 `--list-devices`: print the device list and exit.
    #[arg(long = "list-devices", hide = true)]
    pub list_devices: bool,

    /// b10621 `--gpu-layers` / `--n-gpu-layers`: layers to keep in VRAM.
    #[arg(
        long = "gpu-layers",
        alias = "n-gpu-layers",
        env = "LLAMA_ARG_N_GPU_LAYERS",
        value_name = "N",
        hide = true
    )]
    pub gpu_layers: Option<String>,

    /// b10621 `--main-gpu`: index of the GPU holding intermediate results.
    #[arg(
        long = "main-gpu",
        env = "LLAMA_ARG_MAIN_GPU",
        value_name = "INDEX",
        hide = true
    )]
    pub main_gpu: Option<String>,

    /// b10621 `--split-mode`: none | layer | row | tensor.
    #[arg(
        long = "split-mode",
        env = "LLAMA_ARG_SPLIT_MODE",
        value_name = "{none,layer,row,tensor}",
        hide = true
    )]
    pub split_mode: Option<String>,

    /// b10621 `--tensor-split`: per-GPU offload proportions.
    #[arg(
        long = "tensor-split",
        env = "LLAMA_ARG_TENSOR_SPLIT",
        value_name = "N0,N1,N2,...",
        hide = true
    )]
    pub tensor_split: Option<String>,

    /// b10621 `--rpc`: comma-separated RPC server list.
    #[arg(
        long = "rpc",
        env = "LLAMA_ARG_RPC",
        value_name = "SERVERS",
        hide = true
    )]
    pub rpc: Option<String>,

    /// b10621 `--no-host`: bypass the host buffer.
    #[arg(long = "no-host", hide = true)]
    pub no_host: bool,

    /// b10621 `--op-offload` (positive half of the pair).
    #[arg(long = "op-offload", overrides_with = "no_op_offload", hide = true)]
    pub op_offload: bool,

    /// b10621 `--no-op-offload` (negative half of the pair).
    #[arg(long = "no-op-offload", overrides_with = "op_offload", hide = true)]
    pub no_op_offload: bool,

    /// b10621 `--kv-offload` (positive half of the pair).
    #[arg(long = "kv-offload", overrides_with = "no_kv_offload", hide = true)]
    pub kv_offload: bool,

    /// b10621 `--no-kv-offload` (negative half of the pair).
    #[arg(long = "no-kv-offload", overrides_with = "kv_offload", hide = true)]
    pub no_kv_offload: bool,

    // ── MoE placement ───────────────────────────────────────────────────
    /// b10621 `--cpu-moe`: keep every MoE expert on the CPU.
    #[arg(long = "cpu-moe", hide = true)]
    pub cpu_moe: bool,

    /// b10621 `--n-cpu-moe`: keep the first N layers' experts on the CPU.
    #[arg(
        long = "n-cpu-moe",
        env = "LLAMA_ARG_N_CPU_MOE",
        value_name = "N",
        hide = true
    )]
    pub n_cpu_moe: Option<String>,

    // ── CPU thread pool ─────────────────────────────────────────────────
    /// b10621 `--threads`: generation thread count.
    #[arg(
        long = "threads",
        env = "LLAMA_ARG_THREADS",
        value_name = "N",
        hide = true
    )]
    pub threads: Option<String>,

    /// b10621 `--threads-batch`: prompt-processing thread count.
    #[arg(long = "threads-batch", value_name = "N", hide = true)]
    pub threads_batch: Option<String>,

    /// b10621 `--cpu-mask`: hexadecimal CPU affinity mask.
    #[arg(long = "cpu-mask", value_name = "M", hide = true)]
    pub cpu_mask: Option<String>,

    /// b10621 `--cpu-mask-batch`: affinity mask for batch processing.
    #[arg(long = "cpu-mask-batch", value_name = "M", hide = true)]
    pub cpu_mask_batch: Option<String>,

    /// b10621 `--cpu-range`: CPU range for affinity.
    #[arg(long = "cpu-range", value_name = "lo-hi", hide = true)]
    pub cpu_range: Option<String>,

    /// b10621 `--cpu-range-batch`: CPU range for batch processing.
    #[arg(long = "cpu-range-batch", value_name = "lo-hi", hide = true)]
    pub cpu_range_batch: Option<String>,

    /// b10621 `--cpu-strict`: strict CPU placement.
    #[arg(long = "cpu-strict", value_name = "<0|1>", hide = true)]
    pub cpu_strict: Option<String>,

    /// b10621 `--cpu-strict-batch`: strict placement for batch processing.
    #[arg(long = "cpu-strict-batch", value_name = "<0|1>", hide = true)]
    pub cpu_strict_batch: Option<String>,

    /// b10621 `--poll`: thread-pool polling level.
    #[arg(long = "poll", value_name = "<0...100>", hide = true)]
    pub poll: Option<String>,

    /// b10621 `--poll-batch`: polling level for batch processing.
    #[arg(long = "poll-batch", value_name = "<0|1>", hide = true)]
    pub poll_batch: Option<String>,

    /// b10621 `--prio`: process/thread priority.
    #[arg(long = "prio", value_name = "N", hide = true)]
    pub prio: Option<String>,

    /// b10621 `--prio-batch`: priority for batch processing.
    #[arg(long = "prio-batch", value_name = "N", hide = true)]
    pub prio_batch: Option<String>,

    /// b10621 `--numa`: distribute | isolate | numactl.
    #[arg(
        long = "numa",
        env = "LLAMA_ARG_NUMA",
        value_name = "TYPE",
        hide = true
    )]
    pub numa: Option<String>,

    // ── model metadata and buffers ──────────────────────────────────────
    /// b10621 `--override-kv`: override GGUF metadata by key.
    #[arg(long = "override-kv", value_name = "KEY=TYPE:VALUE,...", hide = true)]
    pub override_kv: Option<String>,

    /// b10621 `--override-tensor`: override a tensor's buffer type.
    #[arg(
        long = "override-tensor",
        env = "LLAMA_ARG_OVERRIDE_TENSOR",
        value_name = "<tensor name pattern>=<buffer type>,...",
        hide = true
    )]
    pub override_tensor: Option<String>,

    // ── context fitting ─────────────────────────────────────────────────
    /// b10621 `--fit`: adjust unset arguments to fit device memory.
    #[arg(
        long = "fit",
        env = "LLAMA_ARG_FIT",
        value_name = "[on|off]",
        hide = true
    )]
    pub fit: Option<String>,

    /// b10621 `--fit-ctx`: minimum context `--fit` may choose.
    #[arg(
        long = "fit-ctx",
        env = "LLAMA_ARG_FIT_CTX",
        value_name = "N",
        hide = true
    )]
    pub fit_ctx: Option<String>,

    /// b10621 `--fit-target`: per-device memory margin for `--fit`.
    #[arg(
        long = "fit-target",
        env = "LLAMA_ARG_FIT_TARGET",
        value_name = "MiB0,MiB1,MiB2,...",
        hide = true
    )]
    pub fit_target: Option<String>,

    // ── KV cache maintenance ────────────────────────────────────────────
    /// b10621 `--defrag-thold`: KV defragmentation threshold (deprecated).
    #[arg(
        long = "defrag-thold",
        env = "LLAMA_ARG_DEFRAG_THOLD",
        value_name = "N",
        hide = true
    )]
    pub defrag_thold: Option<String>,

    // ── profiling ───────────────────────────────────────────────────────
    /// b10621 `--perf` (positive half of the pair).
    #[arg(long = "perf", overrides_with = "no_perf", hide = true)]
    pub perf: bool,

    /// b10621 `--no-perf` (negative half of the pair).
    #[arg(long = "no-perf", overrides_with = "perf", hide = true)]
    pub no_perf: bool,
}

/// One rejected option, ready to render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlCompatRejection {
    /// The option as the operator wrote it, for example `--split-mode`.
    pub option: &'static str,
    /// The requested value, or the flag name again for a value-less flag.
    pub value: String,
    /// Why mlxcel cannot reproduce b10621's behavior for that value.
    pub limitation: &'static str,
    /// What to use instead, or `None` when nothing corresponds.
    pub alternative: Option<&'static str>,
}

impl std::fmt::Display for GgmlCompatRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} is not supported: {}",
            self.option, self.value, self.limitation
        )?;
        match self.alternative {
            Some(alternative) => write!(f, "\nUse instead: {alternative}"),
            None => write!(
                f,
                "\nThere is no mlxcel equivalent; drop the flag to start the server."
            ),
        }
    }
}

impl GgmlCompatArgs {
    /// Reject every value whose b10621 meaning mlxcel cannot reproduce.
    ///
    /// `model_layers` is the loaded checkpoint's transformer layer count, used
    /// only to decide whether a numeric `--gpu-layers` asks for a full offload
    /// (inert, since mlxcel always runs every layer on the accelerator) or a
    /// partial one (unsupported). `None` when it could not be read, which
    /// makes any non-negative count unsupported rather than guessed at.
    ///
    /// # Errors
    ///
    /// Returns the first rejection in a fixed order, so a command line
    /// carrying several unsupported values always reports the same one.
    pub fn ensure_inert(&self, model_layers: Option<usize>) -> Result<(), GgmlCompatRejection> {
        self.rejection(model_layers).map_or(Ok(()), Err)
    }

    /// The subset of [`ensure_inert`](Self::ensure_inert) that needs no model.
    ///
    /// Run this before the model reference is resolved, so `--numa distribute`
    /// or `--rpc host:1` is reported immediately instead of after a multi-
    /// gigabyte download. Only `--gpu-layers` is deferred, because telling a
    /// full offload from a partial one needs the checkpoint's layer count;
    /// [`ensure_inert`](Self::ensure_inert) covers it once that is known.
    ///
    /// # Errors
    ///
    /// Same rejections as [`ensure_inert`](Self::ensure_inert), minus the
    /// `--gpu-layers` arm.
    pub fn ensure_inert_before_model(&self) -> Result<(), GgmlCompatRejection> {
        self.rejection_inner(None, false).map_or(Ok(()), Err)
    }

    /// The first unsupported request in this argument set, if any.
    #[must_use]
    pub fn rejection(&self, model_layers: Option<usize>) -> Option<GgmlCompatRejection> {
        self.rejection_inner(model_layers, true)
    }

    /// Shared body of [`rejection`](Self::rejection) and
    /// [`ensure_inert_before_model`](Self::ensure_inert_before_model).
    ///
    /// `check_gpu_layers` is false only for the pre-resolution pass, where the
    /// layer count is not yet knowable.
    fn rejection_inner(
        &self,
        model_layers: Option<usize>,
        check_gpu_layers: bool,
    ) -> Option<GgmlCompatRejection> {
        const NO_GGML: &str = "mlxcel runs every tensor operation through MLX on one Metal or \
                               CUDA device and has no GGML backend, so there is nothing to place, \
                               offload, or repack";
        const NO_CPU_POOL: &str = "mlxcel has no GGML CPU thread pool to place or size; its CPU \
                                   work is tokenization and HTTP handling, and every tensor \
                                   operation runs on the accelerator";
        const NO_MULTI_DEVICE: &str = "mlxcel drives one accelerator per process and does not \
                                       split a model across devices from this flag";
        const NO_GGUF_METADATA: &str = "mlxcel loads MLX SafeTensors, which carry no GGUF \
                                        key/value metadata block to override";

        // ── loading ─────────────────────────────────────────────────────
        if self.check_tensors {
            return Some(reject(
                "--check-tensors",
                "--check-tensors",
                "mlxcel validates a SafeTensors header's shapes and dtypes while loading but has \
                 no GGML tensor-data scan to run",
                Some("`mlxcel inspect <model>` to report a checkpoint's tensors before serving"),
            ));
        }
        if let Some(mode) = present(self.load_mode.as_deref()) {
            // `auto` and `mmap` both describe what mlxcel already does.
            if !matches!(mode, "auto" | "mmap") {
                return Some(reject_owned(
                    "--load-mode",
                    mode,
                    "mlxcel memory-maps its SafeTensors shards and cannot lock, pin, or \
                     DirectIO-read them; only `auto` and `mmap` describe what it already does",
                    Some("MLXCEL_WIRED_LIMIT to bound Apple Silicon wired memory"),
                ));
            }
        }
        if self.mlock {
            return Some(reject(
                "--mlock",
                "--mlock",
                "mlxcel cannot pin its weights into resident memory; MLX owns the allocation and \
                 the operating system owns the residency policy",
                Some("MLXCEL_WIRED_LIMIT to bound Apple Silicon wired memory"),
            ));
        }
        if self.no_mmap {
            return Some(reject(
                "--no-mmap",
                "--no-mmap",
                "mlxcel always memory-maps its SafeTensors shards and has no read-into-anonymous \
                 loading path",
                None,
            ));
        }
        if self.direct_io {
            return Some(reject(
                "--direct-io",
                "--direct-io",
                "mlxcel reads weights through the page cache and has no DirectIO path",
                None,
            ));
        }

        // ── attention kernel ────────────────────────────────────────────
        if let Some(value) = present(self.flash_attn.as_deref()) {
            let autoy = matches!(value, "auto" | "-1");
            if FALSEY.contains(&value) {
                return Some(reject_owned(
                    "--flash-attn",
                    value,
                    "mlxcel always attends through MLX's fused scaled-dot-product kernel and has \
                     no unfused path to fall back to, so flash attention cannot be turned off",
                    None,
                ));
            }
            if !TRUTHY.contains(&value) && !autoy {
                return Some(reject_owned(
                    "--flash-attn",
                    value,
                    "b10621 accepts only `on`, `off`, or `auto`",
                    Some("`--flash-attn auto`, which is what mlxcel already does"),
                ));
            }
        }

        // ── device placement ────────────────────────────────────────────
        if let Some(device) = present(self.device.as_deref()) {
            return Some(reject_owned(
                "--device",
                device,
                "mlxcel selects its accelerator from the build and the runtime environment, not \
                 from a GGML device list, and cannot disable offloading",
                Some("MLXCEL_DEVICE=gpu|cpu to choose the execution device"),
            ));
        }
        if self.list_devices {
            return Some(reject(
                "--list-devices",
                "--list-devices",
                "mlxcel has no GGML device registry to enumerate; it drives one MLX device",
                Some("`mlxcel inspect <model>` to report the device a model would load on"),
            ));
        }
        if check_gpu_layers
            && let Some(layers) = present(self.gpu_layers.as_deref())
            && let Some(rejection) = gpu_layers_rejection(layers, model_layers)
        {
            return Some(rejection);
        }
        if let Some(index) = present(self.main_gpu.as_deref())
            && index.trim() != "0"
        {
            return Some(reject_owned(
                "--main-gpu",
                index,
                NO_MULTI_DEVICE,
                Some("docs/distributed.md for mlxcel's tensor- and pipeline-parallel setup"),
            ));
        }
        if let Some(mode) = present(self.split_mode.as_deref())
            && mode != "none"
        {
            return Some(reject_owned(
                "--split-mode",
                mode,
                NO_MULTI_DEVICE,
                Some(
                    "--tensor-parallel / --pipeline-parallel; see docs/distributed.md. \
                     `--split-mode none` is accepted as the single-device case",
                ),
            ));
        }
        if let Some(split) = present(self.tensor_split.as_deref()) {
            return Some(reject_owned(
                "--tensor-split",
                split,
                NO_MULTI_DEVICE,
                Some("--tensor-parallel / --pipeline-parallel; see docs/distributed.md"),
            ));
        }
        if let Some(servers) = present(self.rpc.as_deref()) {
            return Some(reject_owned(
                "--rpc",
                servers,
                "mlxcel has no GGML RPC backend; its distributed execution uses its own transport \
                 and node roles",
                Some("--node-role / --peers; see docs/distributed.md"),
            ));
        }
        if self.no_host {
            return Some(reject("--no-host", "--no-host", NO_GGML, None));
        }
        if self.no_op_offload {
            return Some(reject(
                "--no-op-offload",
                "--no-op-offload",
                "mlxcel runs every tensor operation on the accelerator and has no host-execution \
                 fallback to fall back to",
                None,
            ));
        }
        if self.no_kv_offload {
            return Some(reject(
                "--no-kv-offload",
                "--no-kv-offload",
                "mlxcel keeps the KV cache in the accelerator's memory and cannot hold it on the \
                 host",
                Some("--cache-type-k / --cache-type-v to shrink the cache instead"),
            ));
        }

        // ── MoE placement ───────────────────────────────────────────────
        if self.cpu_moe {
            return Some(reject(
                "--cpu-moe",
                "--cpu-moe",
                "mlxcel keeps every expert on the accelerator; there is no CPU expert path to move \
                 them to",
                None,
            ));
        }
        if let Some(count) = present(self.n_cpu_moe.as_deref())
            && count.trim() != "0"
        {
            return Some(reject_owned(
                "--n-cpu-moe",
                count,
                "mlxcel keeps every expert on the accelerator; there is no CPU expert path to move \
                 them to",
                None,
            ));
        }

        // ── CPU thread pool ─────────────────────────────────────────────
        for (option, value, inert) in [
            ("--threads", self.threads.as_deref(), "-1"),
            ("--threads-batch", self.threads_batch.as_deref(), "-1"),
            ("--cpu-strict", self.cpu_strict.as_deref(), "0"),
            ("--cpu-strict-batch", self.cpu_strict_batch.as_deref(), "0"),
            ("--poll", self.poll.as_deref(), "50"),
            ("--poll-batch", self.poll_batch.as_deref(), "50"),
            ("--prio", self.prio.as_deref(), "0"),
            ("--prio-batch", self.prio_batch.as_deref(), "0"),
        ] {
            if let Some(value) = present(value)
                && value.trim() != inert
            {
                return Some(reject_owned(option, value, NO_CPU_POOL, None));
            }
        }
        for (option, value) in [
            ("--cpu-mask", self.cpu_mask.as_deref()),
            ("--cpu-mask-batch", self.cpu_mask_batch.as_deref()),
            ("--cpu-range", self.cpu_range.as_deref()),
            ("--cpu-range-batch", self.cpu_range_batch.as_deref()),
        ] {
            if let Some(value) = present(value) {
                return Some(reject_owned(option, value, NO_CPU_POOL, None));
            }
        }
        if let Some(numa) = present(self.numa.as_deref()) {
            return Some(reject_owned(
                "--numa",
                numa,
                "mlxcel has no NUMA-aware thread placement to configure; the accelerator owns the \
                 memory its tensors live in",
                None,
            ));
        }

        // ── model metadata and buffers ──────────────────────────────────
        if let Some(kv) = present(self.override_kv.as_deref()) {
            return Some(reject_owned(
                "--override-kv",
                kv,
                NO_GGUF_METADATA,
                Some(
                    "edit the checkpoint's config.json, or `mlxcel surgery` for weight-level edits",
                ),
            ));
        }
        if let Some(spec) = present(self.override_tensor.as_deref()) {
            return Some(reject_owned("--override-tensor", spec, NO_GGML, None));
        }

        // ── context fitting ─────────────────────────────────────────────
        if let Some(fit) = present(self.fit.as_deref())
            && !FALSEY.contains(&fit)
        {
            return Some(reject_owned(
                "--fit",
                fit,
                "mlxcel does not resize unset arguments to fit device memory; it estimates the \
                 footprint and refuses to start when it will not fit",
                Some("--estimate-memory, with --ctx-size chosen explicitly"),
            ));
        }
        for (option, value) in [
            ("--fit-ctx", self.fit_ctx.as_deref()),
            ("--fit-target", self.fit_target.as_deref()),
        ] {
            if let Some(value) = present(value) {
                return Some(reject_owned(
                    option,
                    value,
                    "mlxcel has no automatic context fitting for this to bound",
                    Some("--estimate-memory, with --ctx-size chosen explicitly"),
                ));
            }
        }

        // ── KV cache maintenance ────────────────────────────────────────
        if let Some(thold) = present(self.defrag_thold.as_deref()) {
            return Some(reject_owned(
                "--defrag-thold",
                thold,
                "mlxcel's paged KV cache reclaims blocks on release and has no defragmentation \
                 pass to threshold (b10621 deprecates this option too)",
                None,
            ));
        }

        // ── profiling ───────────────────────────────────────────────────
        if self.perf {
            return Some(reject(
                "--perf",
                "--perf",
                "mlxcel has no libllama performance timers to enable; its counters are exposed \
                 over HTTP instead",
                Some("--metrics for the Prometheus endpoint, or --slots for per-slot state"),
            ));
        }

        None
    }

    /// Apply the b10621 environment bindings clap cannot express.
    ///
    /// Value-taking options are bound through clap; the value-less flags and
    /// `--x` / `--no-x` pairs are resolved here so their environment
    /// vocabulary matches b10621's exactly rather than clap's. An explicit
    /// command-line occurrence always wins, which is safe because it can only
    /// turn a flag on.
    ///
    /// # Errors
    ///
    /// Returns the variable name and its value when b10621's
    /// `parse_bool_value` would throw on it, which is what b10621 does.
    pub fn apply_env_bindings(&mut self) -> Result<(), (&'static str, String)> {
        self.backend_sampling |= env_flag("LLAMA_ARG_BACKEND_SAMPLING");
        self.cpu_moe |= env_flag("LLAMA_ARG_CPU_MOE");
        self.no_host |= env_flag("LLAMA_ARG_NO_HOST");
        self.mlock |= env_flag("LLAMA_ARG_MLOCK");

        for (var, positive, negative) in [
            ("LLAMA_ARG_MMAP", &mut self.mmap, &mut self.no_mmap),
            ("LLAMA_ARG_DIO", &mut self.direct_io, &mut self.no_direct_io),
            ("LLAMA_ARG_REPACK", &mut self.repack, &mut self.no_repack),
            ("LLAMA_ARG_PERF", &mut self.perf, &mut self.no_perf),
            (
                "LLAMA_ARG_KV_OFFLOAD",
                &mut self.kv_offload,
                &mut self.no_kv_offload,
            ),
        ] {
            // An explicit command-line occurrence of either half wins.
            if *positive || *negative {
                continue;
            }
            match env_bool_pair(var) {
                None => {}
                Some(Ok(true)) => *positive = true,
                Some(Ok(false)) => *negative = true,
                Some(Err(raw)) => return Err((var, raw)),
            }
        }
        // `--op-offload` has no environment binding in b10621.
        Ok(())
    }
}

/// `Some(trimmed)` when a value is present and not whitespace-only.
///
/// An environment-bound option routinely arrives as an empty string from a
/// shell variable that was never set, and refusing to start over one would
/// make the compatibility surface worse than ignoring the flag. b10621 tests
/// these fields with `.empty()` for the same reason.
fn present(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|v| !v.is_empty())
}

fn reject(
    option: &'static str,
    value: &'static str,
    limitation: &'static str,
    alternative: Option<&'static str>,
) -> GgmlCompatRejection {
    GgmlCompatRejection {
        option,
        value: value.to_owned(),
        limitation,
        alternative,
    }
}

fn reject_owned(
    option: &'static str,
    value: &str,
    limitation: &'static str,
    alternative: Option<&'static str>,
) -> GgmlCompatRejection {
    GgmlCompatRejection {
        option,
        value: value.to_owned(),
        limitation,
        alternative,
    }
}

/// Classify a `--gpu-layers` value.
///
/// mlxcel always runs every layer on the accelerator, so a request for a full
/// offload is inert and a request for a partial one is not. `auto` and `all`
/// are b10621's own spellings for "as many as possible"; a negative count is
/// llama.cpp's historical spelling of the same thing. A non-negative count is
/// inert only when it covers the whole model, which needs the layer count;
/// without it the value is unsupported rather than guessed at.
fn gpu_layers_rejection(value: &str, model_layers: Option<usize>) -> Option<GgmlCompatRejection> {
    const LIMITATION: &str = "mlxcel runs every layer on the accelerator and has no partial \
                              CPU offload, so it cannot honour a layer budget smaller than the \
                              model";
    let trimmed = value.trim();
    if matches!(trimmed, "auto" | "all") {
        return None;
    }
    let Ok(count) = trimmed.parse::<i64>() else {
        return Some(reject_owned(
            "--gpu-layers",
            trimmed,
            "b10621 accepts an exact number, `auto`, or `all`",
            Some("`--gpu-layers all`, or drop the flag: mlxcel always offloads every layer"),
        ));
    };
    if count < 0 {
        // llama.cpp's historical `-ngl -1` means "all layers".
        return None;
    }
    const ALTERNATIVE: &str = "`--gpu-layers all`, or drop the flag: mlxcel always offloads \
                               every layer, so only a value covering the whole model is inert";
    match model_layers {
        Some(layers) if count >= layers as i64 => None,
        _ => Some(reject_owned(
            "--gpu-layers",
            trimmed,
            LIMITATION,
            Some(ALTERNATIVE),
        )),
    }
}

/// The transformer layer count in a checkpoint's `config.json`, if readable.
///
/// Looks at `num_hidden_layers`, then `text_config.num_hidden_layers` for the
/// VLM checkpoints that nest their decoder configuration. Returns `None` for
/// an unreadable or unrecognized config rather than guessing; the caller then
/// treats every non-negative `--gpu-layers` as unsupported.
#[must_use]
pub fn read_model_layer_count(model_path: &Path) -> Option<usize> {
    let content = std::fs::read_to_string(model_path.join("config.json")).ok()?;
    let config = serde_json::from_str::<serde_json::Value>(&content).ok()?;
    config
        .get("num_hidden_layers")
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|text| text.get("num_hidden_layers"))
        })
        .and_then(serde_json::Value::as_u64)
        .map(|n| n as usize)
}

#[cfg(test)]
#[path = "ggml_compat_args_tests.rs"]
mod tests;
