//! CUDA custom-dispatch definition for Gemma3n Q4 prefill projections.
//!
//! The PTX is generated only by the CUDA-IREE build and embedded into every
//! native-QMV MLIR module. Other IREE targets never reference this module.

use std::fmt::Write as _;
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
use std::path::PathBuf;
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
use std::process::Command;

pub(crate) const GEMMA3N_QMV_ABI_VERSION: u32 = 6;
pub(crate) const GEMMA3N_QMV_KERNEL_VERSION: u32 = 8;
pub(crate) const GEMMA3N_QMV_MIN_SM: u32 = 80;

#[cfg(xla_iree_cuda)]
const GEMMA3N_QMV_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemma3n_qmv_sm80.ptx"));

pub(crate) fn is_available() -> bool {
    cfg!(xla_iree_cuda)
}

#[cfg(xla_iree_cuda)]
fn ptx() -> &'static [u8] {
    GEMMA3N_QMV_PTX
}

#[cfg(not(xla_iree_cuda))]
fn ptx() -> &'static [u8] {
    &[]
}

/// Stable metadata hash. PTX bytes themselves are embedded in the MLIR and
/// therefore already participate in the graph/cache hash; this hash exists for
/// compact logs and artifact-identity diagnostics.
pub(crate) fn ptx_fnv1a64() -> u64 {
    ptx().iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

pub(crate) fn artifact_identity() -> String {
    format!(
        "gemma3n-qmv:kernel-v{}:abi-v{}:min-sm{}:ptx-fnv1a64={:016x}",
        GEMMA3N_QMV_KERNEL_VERSION,
        GEMMA3N_QMV_ABI_VERSION,
        GEMMA3N_QMV_MIN_SM,
        ptx_fnv1a64(),
    )
}

pub(crate) fn target_alias() -> &'static str {
    "#gemma3n_qmv_target = #hal.executable.target<\"cuda\", \"cuda-nvptx-fb\">\n"
}

fn sdpa_vector_export() -> &'static str {
    "    hal.executable.export public @gemma3n_sdpa_vector ordinal(5)\n\
     \x20       layout(#hal.pipeline.layout<constants = 6, bindings = [\n\
     \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
     \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
     \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
     \x20         #hal.pipeline.binding<storage_buffer>\n\
     \x20       ]>) count(%device: !hal.device, %query_heads: index) -> \
     (index, index, index) {\n\
     \x20     %c1 = arith.constant 1 : index\n\
     \x20     hal.return %query_heads, %c1, %c1 : index, index, index\n\
     \x20   } attributes {\n\
     \x20     workgroup_size = [1024 : index, 1 : index, 1 : index]\n\
     \x20   }\n"
}

pub(crate) fn executable_source() -> String {
    assert!(
        is_available(),
        "Gemma3n native QMV requires a CUDA-IREE build"
    );
    let bytes = ptx();
    assert!(!bytes.is_empty(), "Gemma3n native QMV PTX is empty");
    let mut dense = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(dense, "{byte:02X}");
    }
    format!(
        "  // {identity}\n  hal.executable.source private @custom_qmv attributes {{\n\
         \x20   objects = #hal.executable.objects<{{\n\
         \x20     #gemma3n_qmv_target = [#hal.executable.object<{{\n\
         \x20       path = \"gemma3n_qmv_compute80.ptx\",\n\
         \x20       data = dense<\"0x{dense}\"> : vector<{byte_len}xi8>\n\
         \x20     }}>]\n\
         \x20   }}>\n\
         \x20 }} {{\n\
         \x20   hal.executable.export public @gemma3n_qmv ordinal(0)\n\
         \x20       layout(#hal.pipeline.layout<constants = 3, bindings = [\n\
         \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
         \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
         \x20         #hal.pipeline.binding<storage_buffer>\n\
         \x20       ]>) count(%device: !hal.device, %n: index, %m: index) -> \
         (index, index, index) {{\n\
         \x20     %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)>()[%n]\n\
         \x20     %c1 = arith.constant 1 : index\n\
         \x20     hal.return %x, %m, %c1 : index, index, index\n\
         \x20   }} attributes {{\n\
         \x20     workgroup_size = [32 : index, 8 : index, 1 : index]\n\
         \x20   }}\n\
         \x20   hal.executable.export public @gemma3n_tanh ordinal(1)\n\
         \x20       layout(#hal.pipeline.layout<constants = 1, bindings = [\n\
         \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
         \x20         #hal.pipeline.binding<storage_buffer>\n\
         \x20       ]>) count(%device: !hal.device, %length: index) -> \
         (index, index, index) {{\n\
         \x20     %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 256)>()[%length]\n\
         \x20     %c1 = arith.constant 1 : index\n\
         \x20     hal.return %x, %c1, %c1 : index, index, index\n\
         \x20   }} attributes {{\n\
         \x20     workgroup_size = [256 : index, 1 : index, 1 : index]\n\
         \x20   }}\n\
         \x20   hal.executable.export public @gemma3n_altup_coeff ordinal(2)\n\
         \x20       layout(#hal.pipeline.layout<constants = 3, bindings = [\n\
         \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
         \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
         \x20         #hal.pipeline.binding<storage_buffer>\n\
         \x20       ]>) count(%device: !hal.device, %rows: index, %output_width: index) -> \
         (index, index, index) {{\n\
         \x20     %length = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%rows, %output_width]\n\
         \x20     %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 256)>()[%length]\n\
         \x20     %c1 = arith.constant 1 : index\n\
         \x20     hal.return %x, %c1, %c1 : index, index, index\n\
         \x20   }} attributes {{\n\
         \x20     workgroup_size = [256 : index, 1 : index, 1 : index]\n\
         \x20   }}\n\
         \x20   hal.executable.export public @gemma3n_altup_predict ordinal(3)\n\
         \x20       layout(#hal.pipeline.layout<constants = 3, bindings = [\n\
         \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
         \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
         \x20         #hal.pipeline.binding<storage_buffer>\n\
         \x20       ]>) count(%device: !hal.device, %plane_count: index, %rows: index, \
         %hidden: index) -> (index, index, index) {{\n\
         \x20     %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%hidden]\n\
         \x20     %c1 = arith.constant 1 : index\n\
         \x20     hal.return %x, %rows, %c1 : index, index, index\n\
         \x20   }} attributes {{\n\
         \x20     workgroup_size = [32 : index, 1 : index, 1 : index]\n\
         \x20   }}\n\
         \x20   hal.executable.export public @gemma3n_geglu_bf16 ordinal(4)\n\
         \x20       layout(#hal.pipeline.layout<constants = 1, bindings = [\n\
         \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
         \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
         \x20         #hal.pipeline.binding<storage_buffer>\n\
         \x20       ]>) count(%device: !hal.device, %length: index) -> \
         (index, index, index) {{\n\
         \x20     %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 256)>()[%length]\n\
         \x20     %c1 = arith.constant 1 : index\n\
         \x20     hal.return %x, %c1, %c1 : index, index, index\n\
         \x20   }} attributes {{\n\
         \x20     workgroup_size = [256 : index, 1 : index, 1 : index]\n\
         \x20   }}\n\
         {sdpa_vector_export}\
         \x20 }}\n",
        identity = artifact_identity(),
        byte_len = bytes.len(),
        sdpa_vector_export = sdpa_vector_export(),
    )
}

#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
fn run_cuda_stablehlo_probe(
    tag: &str,
    mlir: String,
    inputs: &[(&[f32], &[usize])],
    output_len: usize,
) -> Result<Vec<f32>, String> {
    let compiler = std::env::var_os("MLXCEL_XLA_IREE_COMPILE")
        .map(PathBuf::from)
        .or_else(|| option_env!("MLXCEL_XLA_IREE_COMPILE").map(PathBuf::from))
        .ok_or("set MLXCEL_XLA_IREE_COMPILE for the Gemma3n diagnostic")?;
    let runner = std::env::var_os("MLXCEL_XLA_IREE_RUN_MODULE")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("IREE_CUDA_HOME").map(|home| {
                PathBuf::from(home)
                    .join("build")
                    .join("tools")
                    .join("iree-run-module")
            })
        })
        .ok_or("set MLXCEL_XLA_IREE_RUN_MODULE or IREE_CUDA_HOME")?;
    let stem = format!("mlxcel-gemma3n-{tag}-{}", std::process::id());
    let mlir_path = std::env::temp_dir().join(format!("{stem}.mlir"));
    let vmfb_path = std::env::temp_dir().join(format!("{stem}.vmfb"));
    let input_paths = (0..inputs.len())
        .map(|index| std::env::temp_dir().join(format!("{stem}-input{index}.bin")))
        .collect::<Vec<_>>();
    let output_path = std::env::temp_dir().join(format!("{stem}-output.bin"));
    let result = (|| {
        std::fs::write(&mlir_path, mlir)
            .map_err(|error| format!("write {}: {error}", mlir_path.display()))?;
        for ((values, shape), path) in inputs.iter().zip(&input_paths) {
            let expected = shape.iter().product::<usize>();
            if values.len() != expected {
                return Err(format!(
                    "Gemma3n {tag} input has {} values, expected {expected} for {shape:?}",
                    values.len()
                ));
            }
            std::fs::write(
                path,
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .map_err(|error| format!("write {}: {error}", path.display()))?;
        }
        let compiled = Command::new(&compiler)
            .arg("--iree-input-type=stablehlo")
            .arg("--iree-hal-target-device=cuda")
            .arg("--iree-cuda-target=sm_80")
            .arg(&mlir_path)
            .arg("-o")
            .arg(&vmfb_path)
            .output()
            .map_err(|error| format!("run {}: {error}", compiler.display()))?;
        if !compiled.status.success() {
            return Err(format!(
                "iree-compile failed:\n{}",
                String::from_utf8_lossy(&compiled.stderr)
            ));
        }
        let mut command = Command::new(&runner);
        command
            .arg("--device=cuda")
            .arg(format!("--module={}", vmfb_path.display()))
            .arg("--function=main");
        for ((_, shape), path) in inputs.iter().zip(&input_paths) {
            let dimensions = shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("x");
            command.arg(format!("--input={dimensions}xf32=@{}", path.display()));
        }
        let ran = command
            .arg(format!("--output=@{}", output_path.display()))
            .output()
            .map_err(|error| format!("run {}: {error}", runner.display()))?;
        if !ran.status.success() {
            return Err(format!(
                "iree-run-module failed:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&ran.stdout),
                String::from_utf8_lossy(&ran.stderr)
            ));
        }
        let bytes = std::fs::read(&output_path)
            .map_err(|error| format!("read {}: {error}", output_path.display()))?;
        if bytes.len() != output_len * 4 {
            return Err(format!(
                "Gemma3n {tag} output has {} bytes, expected {}",
                bytes.len(),
                output_len * 4
            ));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect())
    })();
    for path in input_paths {
        let _ = std::fs::remove_file(path);
    }
    for path in [mlir_path, vmfb_path, output_path] {
        let _ = std::fs::remove_file(path);
    }
    result
}

/// Run the embedded native QMV against explicit logical BF16-carrier buffers.
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
pub fn run_diagnostic_probe(
    input: &[f32],
    weight: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    if input.len() != m * k || weight.len() != n * k {
        return Err(format!(
            "QMV diagnostic buffer lengths {}/{} do not match {}/{}",
            input.len(),
            weight.len(),
            m * k,
            n * k
        ));
    }
    let mut builder = super::builder::Builder::new().with_gemma3n_qmv(true);
    let input_arg = super::builder::Builder::arg(0, super::builder::Ty::f32(vec![m, k]));
    let weight_arg = super::builder::Builder::arg(1, super::builder::Ty::f32(vec![n, k]));
    let output = builder.gemma3n_qmv(&input_arg, &weight_arg);
    let mlir = format!(
        "{target}module @gemma3n_qmv_probe {{\n{source}  \
         func.func public @main(%arg0: {input_ty}, %arg1: {weight_ty}) -> {result_ty} \
         {{\n{body}    return {result} : {result_ty}\n  }}\n}}\n",
        target = target_alias(),
        source = executable_source(),
        input_ty = input_arg.ty.render(),
        weight_ty = weight_arg.ty.render(),
        result_ty = output.ty.render(),
        body = builder.body(),
        result = output.name,
    );
    run_cuda_stablehlo_probe(
        "real-qmv",
        mlir,
        &[(input, &[m, k]), (weight, &[n, k])],
        m * n,
    )
}

/// Run the Gemma3n BF16 RMSNorm graph against explicit logical carriers.
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
pub fn run_rms_diagnostic_probe(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<Vec<f32>, String> {
    if input.len() != rows * width || weight.len() != width {
        return Err(format!(
            "RMSNorm diagnostic buffer lengths {}/{} do not match {}/{}",
            input.len(),
            weight.len(),
            rows * width,
            width
        ));
    }
    let mut builder = super::builder::Builder::new();
    let input_arg = super::builder::Builder::arg(0, super::builder::Ty::f32(vec![rows, width]));
    let weight_arg = super::builder::Builder::arg(1, super::builder::Ty::f32(vec![width]));
    let eps = builder.const_f32(eps);
    let zero = builder.const_f32(0.0);
    let output = super::gemma3n_emit_ops::rms_last_bf16(
        &mut builder,
        &input_arg,
        Some(&weight_arg),
        &eps,
        &zero,
    );
    let mlir = format!(
        "module @gemma3n_rms_probe {{\n  \
         func.func public @main(%arg0: {input_ty}, %arg1: {weight_ty}) -> {result_ty} \
         {{\n{body}    return {result} : {result_ty}\n  }}\n}}\n",
        input_ty = input_arg.ty.render(),
        weight_ty = weight_arg.ty.render(),
        result_ty = output.ty.render(),
        body = builder.body(),
        result = output.name,
    );
    run_cuda_stablehlo_probe(
        "real-rms",
        mlir,
        &[(input, &[rows, width]), (weight, &[width])],
        rows * width,
    )
}

/// Return the layer attention stages while consuming the exact MLX
/// q-projection, K/V cache, mask, active prediction, and LAUREL carriers.
///
/// Keeping every stage input except q-norm/RoPE identical lets the canonical
/// diagnostic distinguish the first q rotation mismatch from attention math,
/// output projection, post-attention norm, and residual rounding.
#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
pub fn run_attention_diagnostic_probe(
    q_projection: &[f32],
    q_norm_weight: &[f32],
    keys: &[f32],
    values: &[f32],
    mask: &[f32],
    o_weight: &[f32],
    post_norm_weight: &[f32],
    active_prediction: &[f32],
    laurel: &[f32],
    rows: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_base: f64,
    eps: f32,
) -> Result<Vec<f32>, String> {
    let hidden = query_heads * head_dim;
    if query_heads == 0
        || kv_heads == 0
        || !query_heads.is_multiple_of(kv_heads)
        || q_projection.len() != rows * hidden
        || q_norm_weight.len() != head_dim
        || keys.len() != rows * kv_heads * head_dim
        || values.len() != rows * kv_heads * head_dim
        || mask.len() != rows * rows
        || o_weight.len() != hidden * hidden
        || post_norm_weight.len() != hidden
        || active_prediction.len() != rows * hidden
        || laurel.len() != rows * hidden
    {
        return Err("attention diagnostic input lengths do not match the shape".to_string());
    }

    let mut builder = super::builder::Builder::new().with_gemma3n_qmv(true);
    let q_projection_arg =
        super::builder::Builder::arg(0, super::builder::Ty::f32(vec![rows, hidden]));
    let q_norm_arg = super::builder::Builder::arg(1, super::builder::Ty::f32(vec![head_dim]));
    let keys_arg =
        super::builder::Builder::arg(2, super::builder::Ty::f32(vec![rows, kv_heads, head_dim]));
    let values_arg =
        super::builder::Builder::arg(3, super::builder::Ty::f32(vec![rows, kv_heads, head_dim]));
    let mask_arg = super::builder::Builder::arg(4, super::builder::Ty::f32(vec![rows, rows]));
    let o_weight_arg =
        super::builder::Builder::arg(5, super::builder::Ty::f32(vec![hidden, hidden]));
    let post_norm_arg = super::builder::Builder::arg(6, super::builder::Ty::f32(vec![hidden]));
    let active_arg = super::builder::Builder::arg(7, super::builder::Ty::f32(vec![rows, hidden]));
    let laurel_arg = super::builder::Builder::arg(8, super::builder::Ty::f32(vec![rows, hidden]));

    let zero = builder.const_f32(0.0);
    let eps_value = builder.const_f32(eps);
    let q = builder.reshape(&q_projection_arg, vec![rows, query_heads, head_dim]);
    let q = super::gemma3n_emit_ops::rms_last_bf16(
        &mut builder,
        &q,
        Some(&q_norm_arg),
        &eps_value,
        &zero,
    );
    let positions = builder.iota(rows);
    let (cosine, sine) = super::rope::rope_tables_from_inv(
        &super::rope::plain_inv_freq_with_base(head_dim, rope_base),
        head_dim,
        rows,
        false,
    );
    let cosine = builder.const_tensor_f32(&cosine, vec![rows, head_dim]);
    let sine = builder.const_tensor_f32(&sine, vec![rows, head_dim]);
    let q = super::gemma3n_emit_ops::rope(
        &mut builder,
        &q,
        &positions,
        &cosine,
        &sine,
        rows,
        query_heads,
        head_dim,
    );

    let group = query_heads / kv_heads;
    let grouped_q = builder.reshape(&q, vec![rows, kv_heads, group, head_dim]);
    let raw_scores = builder.dot_general(
        &grouped_q,
        &keys_arg,
        &[1],
        &[1],
        &[3],
        &[2],
        vec![kv_heads, rows, group, rows],
    );
    let raw_scores = super::gemma3n_emit_ops::round_bf16(&mut builder, &raw_scores);
    let bf16_mask = super::gemma3n_emit_ops::round_bf16(&mut builder, &mask_arg);
    let bf16_mask = builder.broadcast(&bf16_mask, &[1, 3], vec![kv_heads, rows, group, rows]);
    let masked_scores = builder.add(&raw_scores, &bf16_mask);
    let masked_scores = super::gemma3n_emit_ops::round_bf16(&mut builder, &masked_scores);
    let neg_inf = builder.const_f32(f32::NEG_INFINITY);
    let probabilities = {
        let scores = &masked_scores;
        let shape = scores.ty.shape.clone();
        let max = builder.reduce_max(scores, 3, &neg_inf);
        let max = builder.broadcast(&max, &[0, 1, 2], shape.clone());
        let shifted = builder.subtract(scores, &max);
        let exp = builder.exponential(&shifted);
        let sum = builder.reduce_add(&exp, 3, &zero);
        let sum = builder.broadcast(&sum, &[0, 1, 2], shape);
        builder.divide(&exp, &sum)
    };
    let probabilities = super::gemma3n_emit_ops::round_bf16(&mut builder, &probabilities);
    let context = builder.dot_general(
        &probabilities,
        &values_arg,
        &[0],
        &[1],
        &[3],
        &[0],
        vec![kv_heads, rows, group, head_dim],
    );
    let context = builder.transpose(&context, &[1, 0, 2, 3]);
    let context = builder.reshape(&context, vec![rows, hidden]);
    let context = super::gemma3n_emit_ops::round_bf16(&mut builder, &context);
    let projected = builder.gemma3n_qmv(&context, &o_weight_arg);
    let post_norm = super::gemma3n_emit_ops::rms_last_bf16(
        &mut builder,
        &projected,
        Some(&post_norm_arg),
        &eps_value,
        &zero,
    );
    let residual = builder.add(&active_arg, &post_norm);
    let residual = super::gemma3n_emit_ops::round_bf16(&mut builder, &residual);
    let with_laurel = builder.add(&residual, &laurel_arg);
    let with_laurel = super::gemma3n_emit_ops::round_bf16(&mut builder, &with_laurel);
    let inv_sqrt2 =
        super::gemma3n_emit_ops::bf16_scalar(&mut builder, std::f32::consts::FRAC_1_SQRT_2);
    let inv_sqrt2 = builder.broadcast(&inv_sqrt2, &[], vec![rows, hidden]);
    let attn_laurel = builder.multiply(&with_laurel, &inv_sqrt2);
    let attn_laurel = super::gemma3n_emit_ops::round_bf16(&mut builder, &attn_laurel);
    let output = flatten_and_concatenate(
        &mut builder,
        &[
            q,
            raw_scores,
            masked_scores,
            probabilities,
            context,
            projected,
            post_norm,
            residual,
            attn_laurel,
        ],
    );

    let mlir = format!(
        "{target}module @gemma3n_attention_probe {{\n{source}  \
         func.func public @main(%arg0: {q_projection_ty}, %arg1: {q_norm_ty}, \
         %arg2: {keys_ty}, %arg3: {values_ty}, %arg4: {mask_ty}, \
         %arg5: {o_weight_ty}, %arg6: {post_norm_ty}, %arg7: {active_ty}, \
         %arg8: {laurel_ty}) -> {result_ty} {{\n{body}    return {result} : \
         {result_ty}\n  }}\n}}\n",
        target = target_alias(),
        source = executable_source(),
        q_projection_ty = q_projection_arg.ty.render(),
        q_norm_ty = q_norm_arg.ty.render(),
        keys_ty = keys_arg.ty.render(),
        values_ty = values_arg.ty.render(),
        mask_ty = mask_arg.ty.render(),
        o_weight_ty = o_weight_arg.ty.render(),
        post_norm_ty = post_norm_arg.ty.render(),
        active_ty = active_arg.ty.render(),
        laurel_ty = laurel_arg.ty.render(),
        result_ty = output.ty.render(),
        body = builder.body(),
        result = output.name,
    );
    run_cuda_stablehlo_probe(
        "attention-stages",
        mlir,
        &[
            (q_projection, &[rows, hidden]),
            (q_norm_weight, &[head_dim]),
            (keys, &[rows, kv_heads, head_dim]),
            (values, &[rows, kv_heads, head_dim]),
            (mask, &[rows, rows]),
            (o_weight, &[hidden, hidden]),
            (post_norm_weight, &[hidden]),
            (active_prediction, &[rows, hidden]),
            (laurel, &[rows, hidden]),
        ],
        rows * hidden * 6 + kv_heads * rows * group * rows * 3,
    )
}

/// Run only the production D=256 MLX vector-SDPA context dispatch.
///
/// This smaller diagnostics-only entry point supports synthetic address,
/// boundary-length, GQA, and rounding fixtures without compiling the Gemma3n
/// output projection tail.
#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
pub fn run_sdpa_vector_context_diagnostic_probe(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    position: usize,
    capacity: usize,
    query_heads: usize,
    kv_heads: usize,
    scale: f32,
    sliding_window: Option<usize>,
) -> Result<Vec<f32>, String> {
    const HEAD_DIM: usize = 256;
    let hidden = query_heads
        .checked_mul(HEAD_DIM)
        .ok_or("SDPA diagnostic hidden width overflow")?;
    let kv_length = capacity
        .checked_mul(kv_heads)
        .and_then(|length| length.checked_mul(HEAD_DIM))
        .ok_or("SDPA diagnostic KV length overflow")?;
    if query_heads == 0
        || kv_heads == 0
        || !query_heads.is_multiple_of(kv_heads)
        || position >= capacity
        || position >= 1024
        || query.len() != hidden
        || keys.len() != kv_length
        || values.len() != kv_length
    {
        return Err("SDPA vector context diagnostic inputs do not match the shape".to_string());
    }

    let mut builder = super::builder::Builder::new().with_gemma3n_qmv(true);
    let query_arg =
        super::builder::Builder::arg(0, super::builder::Ty::f32(vec![query_heads, HEAD_DIM]));
    let keys_arg = super::builder::Builder::arg(
        1,
        super::builder::Ty::f32(vec![capacity, kv_heads, HEAD_DIM]),
    );
    let values_arg = super::builder::Builder::arg(
        2,
        super::builder::Ty::f32(vec![capacity, kv_heads, HEAD_DIM]),
    );
    let position_value = builder.const_i32(position as i32);
    let context = builder.gemma3n_sdpa_vector(
        &query_arg,
        &keys_arg,
        &values_arg,
        &position_value,
        sliding_window,
        scale,
    );
    let output = builder.reshape(&context, vec![hidden]);
    let mlir = format!(
        "{target}module @gemma3n_sdpa_vector_context_probe {{\n{source}  \
         func.func public @main(%arg0: {query_ty}, %arg1: {keys_ty}, \
         %arg2: {values_ty}) -> {result_ty} {{\n\
         {body}    return {result} : {result_ty}\n  }}\n}}\n",
        target = target_alias(),
        source = executable_source(),
        query_ty = query_arg.ty.render(),
        keys_ty = keys_arg.ty.render(),
        values_ty = values_arg.ty.render(),
        result_ty = output.ty.render(),
        body = builder.body(),
        result = output.name,
    );
    run_cuda_stablehlo_probe(
        "sdpa-vector-context",
        mlir,
        &[
            (query, &[query_heads, HEAD_DIM]),
            (keys, &[capacity, kv_heads, HEAD_DIM]),
            (values, &[capacity, kv_heads, HEAD_DIM]),
        ],
        hidden,
    )
}

/// Trace the concrete layer-0 production decode attention stages while
/// consuming exact MLX normalized-input and prefix-KV carriers.
///
/// This wrapper calls the same traced core as normal and ragged decode; it does
/// not carry a separately-authored materialized attention implementation.
#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
pub fn run_decode_attention_diagnostic_probe(
    normalized: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    v_weight: &[f32],
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    prefix_keys: &[f32],
    prefix_values: &[f32],
    o_weight: &[f32],
    post_norm_weight: &[f32],
    active_prediction: &[f32],
    laurel: &[f32],
    position: usize,
    capacity: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_base: f64,
    eps: f32,
    sliding_window: Option<usize>,
) -> Result<Vec<f32>, String> {
    run_production_decode_attention_trace(
        normalized,
        q_weight,
        k_weight,
        v_weight,
        q_norm_weight,
        k_norm_weight,
        prefix_keys,
        prefix_values,
        o_weight,
        post_norm_weight,
        active_prediction,
        laurel,
        position,
        capacity,
        query_heads,
        kv_heads,
        head_dim,
        rope_base,
        eps,
        sliding_window,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
fn run_production_decode_attention_trace(
    normalized: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    v_weight: &[f32],
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    prefix_keys: &[f32],
    prefix_values: &[f32],
    o_weight: &[f32],
    post_norm_weight: &[f32],
    active_prediction: &[f32],
    laurel: &[f32],
    position: usize,
    capacity: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_base: f64,
    eps: f32,
    sliding_window: Option<usize>,
) -> Result<Vec<f32>, String> {
    use super::gemma3n::{Gemma3nConfig, Gemma3nLayerType};
    use super::gemma3n_emit_ops::{
        LayerWeights, attention_decode_traced, bf16_scalar, constants, rms_last_bf16, round_bf16,
    };

    let hidden = query_heads
        .checked_mul(head_dim)
        .ok_or("decode attention hidden width overflow")?;
    let kv_width = kv_heads
        .checked_mul(head_dim)
        .ok_or("decode attention KV width overflow")?;
    let valid_len = position
        .checked_add(1)
        .ok_or("decode attention live length overflow")?;
    if query_heads == 0
        || kv_heads == 0
        || !query_heads.is_multiple_of(kv_heads)
        || valid_len > capacity
        || normalized.len() != hidden
        || q_weight.len() != hidden * hidden
        || k_weight.len() != kv_width * hidden
        || v_weight.len() != kv_width * hidden
        || q_norm_weight.len() != head_dim
        || k_norm_weight.len() != head_dim
        || prefix_keys.len() != position * kv_width
        || prefix_values.len() != position * kv_width
        || o_weight.len() != hidden * hidden
        || post_norm_weight.len() != hidden
        || active_prediction.len() != hidden
        || laurel.len() != hidden
    {
        return Err("production decode attention trace inputs do not match the shape".to_string());
    }

    let mut builder = super::builder::Builder::new().with_gemma3n_qmv(true);
    let normalized_arg = super::builder::Builder::arg(0, super::builder::Ty::f32(vec![1, hidden]));
    let q_weight_arg =
        super::builder::Builder::arg(1, super::builder::Ty::f32(vec![hidden, hidden]));
    let k_weight_arg =
        super::builder::Builder::arg(2, super::builder::Ty::f32(vec![kv_width, hidden]));
    let v_weight_arg =
        super::builder::Builder::arg(3, super::builder::Ty::f32(vec![kv_width, hidden]));
    let q_norm_arg = super::builder::Builder::arg(4, super::builder::Ty::f32(vec![head_dim]));
    let k_norm_arg = super::builder::Builder::arg(5, super::builder::Ty::f32(vec![head_dim]));
    let prefix_keys_arg = super::builder::Builder::arg(
        6,
        super::builder::Ty::f32(vec![position, kv_heads, head_dim]),
    );
    let prefix_values_arg = super::builder::Builder::arg(
        7,
        super::builder::Ty::f32(vec![position, kv_heads, head_dim]),
    );
    let o_weight_arg =
        super::builder::Builder::arg(8, super::builder::Ty::f32(vec![hidden, hidden]));
    let post_norm_arg = super::builder::Builder::arg(9, super::builder::Ty::f32(vec![hidden]));
    let active_arg = super::builder::Builder::arg(10, super::builder::Ty::f32(vec![1, hidden]));
    let laurel_arg = super::builder::Builder::arg(11, super::builder::Ty::f32(vec![1, hidden]));

    let layer_type = if sliding_window.is_some() {
        Gemma3nLayerType::Sliding
    } else {
        Gemma3nLayerType::Full
    };
    let config = Gemma3nConfig {
        context_capacity: capacity,
        max_position_embeddings: capacity.max(1),
        hidden,
        intermediate: vec![hidden],
        n_layers: 1,
        n_q: query_heads,
        n_kv: kv_heads,
        head_dim,
        eps,
        vocab: 1,
        per_layer_vocab: 1,
        hidden_per_layer_input: 1,
        layer_types: vec![layer_type],
        activation_sparsity: vec![0.0],
        sliding_window: sliding_window.unwrap_or(capacity.max(1)),
        rope_theta: rope_base,
        rope_local_base: rope_base,
        final_logit_softcap: None,
        num_kv_shared_layers: 0,
        altup_num_inputs: 2,
        altup_active_idx: 0,
        altup_coef_clip: None,
        altup_correct_scale: false,
        laurel_rank: 1,
        tie_word_embeddings: true,
        quantization: None,
    };
    let constants = constants(&mut builder, &config);
    let unused = normalized_arg.clone();
    let layer = LayerWeights {
        correct_scale: unused.clone(),
        correction: unused.clone(),
        router: unused.clone(),
        router_norm: unused.clone(),
        prediction: unused.clone(),
        laurel_left: unused.clone(),
        laurel_right: unused.clone(),
        laurel_norm: unused.clone(),
        input_norm: unused.clone(),
        post_attn_norm: post_norm_arg.clone(),
        pre_ff_norm: unused.clone(),
        post_ff_norm: unused.clone(),
        wq: q_weight_arg.clone(),
        wk: Some(k_weight_arg.clone()),
        wv: Some(v_weight_arg.clone()),
        wo: o_weight_arg.clone(),
        q_norm: q_norm_arg.clone(),
        k_norm: Some(k_norm_arg.clone()),
        gate: unused.clone(),
        up: unused.clone(),
        down: unused.clone(),
        ple_gate: unused.clone(),
        ple_projection: unused.clone(),
        ple_norm: unused,
    };

    let zero = builder.const_f32(0.0);
    let padding = capacity - position;
    let key_padding = builder.broadcast(&zero, &[], vec![padding, kv_heads, head_dim]);
    let value_padding = builder.broadcast(&zero, &[], vec![padding, kv_heads, head_dim]);
    let initial_keys = builder.concatenate(&prefix_keys_arg, &key_padding, 0);
    let initial_values = builder.concatenate(&prefix_values_arg, &value_padding, 0);
    let mut kcache = builder.reshape(&initial_keys, vec![1, capacity, kv_heads, head_dim]);
    let mut vcache = builder.reshape(&initial_values, vec![1, capacity, kv_heads, head_dim]);
    let position_value = builder.const_i32(position as i32);
    let (projected, trace) = attention_decode_traced(
        &mut builder,
        &normalized_arg,
        &position_value,
        &layer,
        0,
        0,
        &config,
        &constants,
        &mut kcache,
        &mut vcache,
    );
    let valid_keys = builder.slice(&trace.keys, &[(0, valid_len), (0, kv_heads), (0, head_dim)]);
    let valid_values = builder.slice(
        &trace.values,
        &[(0, valid_len), (0, kv_heads), (0, head_dim)],
    );
    let post_norm = rms_last_bf16(
        &mut builder,
        &projected,
        Some(&post_norm_arg),
        &constants.eps,
        &constants.zero,
    );
    let residual = builder.add(&active_arg, &post_norm);
    let residual = round_bf16(&mut builder, &residual);
    let with_laurel = builder.add(&residual, &laurel_arg);
    let with_laurel = round_bf16(&mut builder, &with_laurel);
    let inv_sqrt2 = bf16_scalar(&mut builder, std::f32::consts::FRAC_1_SQRT_2);
    let inv_sqrt2 = builder.broadcast(&inv_sqrt2, &[], vec![1, hidden]);
    let attn_laurel = builder.multiply(&with_laurel, &inv_sqrt2);
    let attn_laurel = round_bf16(&mut builder, &attn_laurel);
    let output = flatten_and_concatenate(
        &mut builder,
        &[
            trace.q_projection,
            trace.q_norm,
            trace.q_rope,
            trace
                .k_projection
                .expect("concrete diagnostic K projection"),
            trace.k_norm.expect("concrete diagnostic K norm"),
            trace.k_rope.expect("concrete diagnostic K RoPE"),
            trace
                .v_projection
                .expect("concrete diagnostic V projection"),
            trace.v_norm.expect("concrete diagnostic V norm"),
            valid_keys,
            valid_values,
            trace.context,
            projected,
            post_norm,
            residual,
            attn_laurel,
        ],
    );

    let mlir = format!(
        "{target}module @gemma3n_production_decode_attention_trace {{\n{source}  \
         func.func public @main(%arg0: {normalized_ty}, %arg1: {q_weight_ty}, \
         %arg2: {k_weight_ty}, %arg3: {v_weight_ty}, %arg4: {q_norm_ty}, \
         %arg5: {k_norm_ty}, %arg6: {prefix_keys_ty}, %arg7: {prefix_values_ty}, \
         %arg8: {o_weight_ty}, %arg9: {post_norm_ty}, %arg10: {active_ty}, \
         %arg11: {laurel_ty}) -> {result_ty} {{\n{body}    return {result} : \
         {result_ty}\n  }}\n}}\n",
        target = target_alias(),
        source = executable_source(),
        normalized_ty = normalized_arg.ty.render(),
        q_weight_ty = q_weight_arg.ty.render(),
        k_weight_ty = k_weight_arg.ty.render(),
        v_weight_ty = v_weight_arg.ty.render(),
        q_norm_ty = q_norm_arg.ty.render(),
        k_norm_ty = k_norm_arg.ty.render(),
        prefix_keys_ty = prefix_keys_arg.ty.render(),
        prefix_values_ty = prefix_values_arg.ty.render(),
        o_weight_ty = o_weight_arg.ty.render(),
        post_norm_ty = post_norm_arg.ty.render(),
        active_ty = active_arg.ty.render(),
        laurel_ty = laurel_arg.ty.render(),
        result_ty = output.ty.render(),
        body = builder.body(),
        result = output.name,
    );
    run_cuda_stablehlo_probe(
        "production-decode-attention-trace",
        mlir,
        &[
            (normalized, &[1, hidden]),
            (q_weight, &[hidden, hidden]),
            (k_weight, &[kv_width, hidden]),
            (v_weight, &[kv_width, hidden]),
            (q_norm_weight, &[head_dim]),
            (k_norm_weight, &[head_dim]),
            (prefix_keys, &[position, kv_heads, head_dim]),
            (prefix_values, &[position, kv_heads, head_dim]),
            (o_weight, &[hidden, hidden]),
            (post_norm_weight, &[hidden]),
            (active_prediction, &[1, hidden]),
            (laurel, &[1, hidden]),
        ],
        hidden * 8 + kv_width * 5 + valid_len * kv_width * 2,
    )
}

/// Return the exact production post-attention MLP stages while consuming the
/// already-proven MLX attention+LAUREL carrier.
///
/// This probe intentionally duplicates the production StableHLO schedule
/// instead of changing the diagnostic layout of the canonical graph. That
/// keeps the bisect isolated from production graph fusion and lets the caller
/// identify the first divergent BF16 boundary.
#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
pub fn run_post_attention_diagnostic_probe(
    attention_laurel: &[f32],
    pre_ff_norm_weight: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    post_ff_norm_weight: &[f32],
    rows: usize,
    hidden: usize,
    intermediate: usize,
    eps: f32,
    sparsity: f32,
) -> Result<Vec<f32>, String> {
    if attention_laurel.len() != rows * hidden
        || pre_ff_norm_weight.len() != hidden
        || gate_weight.len() != intermediate * hidden
        || up_weight.len() != intermediate * hidden
        || down_weight.len() != hidden * intermediate
        || post_ff_norm_weight.len() != hidden
    {
        return Err(
            "post-attention diagnostic input lengths do not match the requested shape".to_string(),
        );
    }

    let mut builder = super::builder::Builder::new().with_gemma3n_qmv(true);
    let attention_arg =
        super::builder::Builder::arg(0, super::builder::Ty::f32(vec![rows, hidden]));
    let pre_ff_norm_arg = super::builder::Builder::arg(1, super::builder::Ty::f32(vec![hidden]));
    let gate_weight_arg =
        super::builder::Builder::arg(2, super::builder::Ty::f32(vec![intermediate, hidden]));
    let up_weight_arg =
        super::builder::Builder::arg(3, super::builder::Ty::f32(vec![intermediate, hidden]));
    let down_weight_arg =
        super::builder::Builder::arg(4, super::builder::Ty::f32(vec![hidden, intermediate]));
    let post_ff_norm_arg = super::builder::Builder::arg(5, super::builder::Ty::f32(vec![hidden]));

    let zero = builder.const_f32(0.0);
    let one = builder.const_f32(1.0);
    let eps_value = builder.const_f32(eps);
    let pre_ff_norm = super::gemma3n_emit_ops::rms_last_bf16(
        &mut builder,
        &attention_arg,
        Some(&pre_ff_norm_arg),
        &eps_value,
        &zero,
    );
    let gate = builder.gemma3n_qmv(&pre_ff_norm, &gate_weight_arg);
    let up = builder.gemma3n_qmv(&pre_ff_norm, &up_weight_arg);

    let sparse =
        super::gemma3n_emit_ops::sparse_gelu_stages(&mut builder, &gate, sparsity, &zero, &one);
    let product = builder.multiply(&sparse.activated, &up);
    let product = super::gemma3n_emit_ops::round_bf16(&mut builder, &product);
    let down = builder.gemma3n_qmv(&product, &down_weight_arg);
    let post_ff_norm = super::gemma3n_emit_ops::rms_last_bf16(
        &mut builder,
        &down,
        Some(&post_ff_norm_arg),
        &eps_value,
        &zero,
    );
    let ff_residual = builder.add(&attention_arg, &post_ff_norm);
    let ff_residual = super::gemma3n_emit_ops::round_bf16(&mut builder, &ff_residual);

    let output = flatten_and_concatenate(
        &mut builder,
        &[
            pre_ff_norm,
            gate,
            up,
            sparse.mean,
            sparse.variance,
            sparse.stddev,
            sparse.cutoff,
            sparse.shifted_raw,
            sparse.shifted,
            sparse.erf,
            sparse.activated,
            product,
            down,
            post_ff_norm,
            ff_residual,
        ],
    );

    let mlir = format!(
        "{target}module @gemma3n_post_attention_probe {{\n{source}  \
         func.func public @main(%arg0: {attention_ty}, %arg1: {pre_ff_norm_ty}, \
         %arg2: {gate_weight_ty}, %arg3: {up_weight_ty}, \
         %arg4: {down_weight_ty}, %arg5: {post_ff_norm_ty}) -> {result_ty} \
         {{\n{body}    return {result} : \
         {result_ty}\n  }}\n}}\n",
        target = target_alias(),
        source = executable_source(),
        attention_ty = attention_arg.ty.render(),
        pre_ff_norm_ty = pre_ff_norm_arg.ty.render(),
        gate_weight_ty = gate_weight_arg.ty.render(),
        up_weight_ty = up_weight_arg.ty.render(),
        down_weight_ty = down_weight_arg.ty.render(),
        post_ff_norm_ty = post_ff_norm_arg.ty.render(),
        result_ty = output.ty.render(),
        body = builder.body(),
        result = output.name,
    );
    let output_len = rows * hidden * 4 + rows * intermediate * 7 + rows * 4;
    run_cuda_stablehlo_probe(
        "post-attention-stages",
        mlir,
        &[
            (attention_laurel, &[rows, hidden]),
            (pre_ff_norm_weight, &[hidden]),
            (gate_weight, &[intermediate, hidden]),
            (up_weight, &[intermediate, hidden]),
            (down_weight, &[hidden, intermediate]),
            (post_ff_norm_weight, &[hidden]),
        ],
        output_len,
    )
}

/// Pin the production native dense-GeGLU schedule from an exact MLX input
/// carrier without expanding the full-graph diagnostic result.
#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
pub fn run_dense_mlp_diagnostic_probe(
    attention_laurel: &[f32],
    pre_ff_norm_weight: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    post_ff_norm_weight: &[f32],
    rows: usize,
    hidden: usize,
    intermediate: usize,
    eps: f32,
) -> Result<Vec<f32>, String> {
    if attention_laurel.len() != rows * hidden
        || pre_ff_norm_weight.len() != hidden
        || gate_weight.len() != intermediate * hidden
        || up_weight.len() != intermediate * hidden
        || down_weight.len() != hidden * intermediate
        || post_ff_norm_weight.len() != hidden
    {
        return Err("dense MLP diagnostic inputs do not match the shape".to_string());
    }

    let mut builder = super::builder::Builder::new().with_gemma3n_qmv(true);
    let attention_arg =
        super::builder::Builder::arg(0, super::builder::Ty::f32(vec![rows, hidden]));
    let pre_ff_norm_arg = super::builder::Builder::arg(1, super::builder::Ty::f32(vec![hidden]));
    let gate_weight_arg =
        super::builder::Builder::arg(2, super::builder::Ty::f32(vec![intermediate, hidden]));
    let up_weight_arg =
        super::builder::Builder::arg(3, super::builder::Ty::f32(vec![intermediate, hidden]));
    let down_weight_arg =
        super::builder::Builder::arg(4, super::builder::Ty::f32(vec![hidden, intermediate]));
    let post_ff_norm_arg = super::builder::Builder::arg(5, super::builder::Ty::f32(vec![hidden]));

    let zero = builder.const_f32(0.0);
    let eps_value = builder.const_f32(eps);
    let pre_ff_norm = super::gemma3n_emit_ops::rms_last_bf16(
        &mut builder,
        &attention_arg,
        Some(&pre_ff_norm_arg),
        &eps_value,
        &zero,
    );
    let gate = builder.gemma3n_qmv(&pre_ff_norm, &gate_weight_arg);
    let up = builder.gemma3n_qmv(&pre_ff_norm, &up_weight_arg);

    let product = super::gemma3n_emit_ops::geglu(&mut builder, &gate, &up);
    let down = builder.gemma3n_qmv(&product, &down_weight_arg);
    let post = super::gemma3n_emit_ops::rms_last_bf16(
        &mut builder,
        &down,
        Some(&post_ff_norm_arg),
        &eps_value,
        &zero,
    );
    let residual = builder.add(&attention_arg, &post);
    let residual = super::gemma3n_emit_ops::round_bf16(&mut builder, &residual);

    let output = flatten_and_concatenate(
        &mut builder,
        &[pre_ff_norm, gate, up, product, down, post, residual],
    );
    let mlir = format!(
        "{target}module @gemma3n_dense_mlp_probe {{\n{source}  \
         func.func public @main(%arg0: {attention_ty}, %arg1: {pre_ff_norm_ty}, \
         %arg2: {gate_weight_ty}, %arg3: {up_weight_ty}, \
         %arg4: {down_weight_ty}, %arg5: {post_ff_norm_ty}) -> {result_ty} \
         {{\n{body}    return {result} : {result_ty}\n  }}\n}}\n",
        target = target_alias(),
        source = executable_source(),
        attention_ty = attention_arg.ty.render(),
        pre_ff_norm_ty = pre_ff_norm_arg.ty.render(),
        gate_weight_ty = gate_weight_arg.ty.render(),
        up_weight_ty = up_weight_arg.ty.render(),
        down_weight_ty = down_weight_arg.ty.render(),
        post_ff_norm_ty = post_ff_norm_arg.ty.render(),
        result_ty = output.ty.render(),
        body = builder.body(),
        result = output.name,
    );
    run_cuda_stablehlo_probe(
        "dense-mlp-stages",
        mlir,
        &[
            (attention_laurel, &[rows, hidden]),
            (pre_ff_norm_weight, &[hidden]),
            (gate_weight, &[intermediate, hidden]),
            (up_weight, &[intermediate, hidden]),
            (down_weight, &[hidden, intermediate]),
            (post_ff_norm_weight, &[hidden]),
        ],
        rows * hidden * 4 + rows * intermediate * 3,
    )
}

/// Isolate AltUp correction by consuming the exact MLX activated carrier.
#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
pub fn run_altup_correct_diagnostic_probe(
    predicted_stacked: &[f32],
    active_prediction: &[f32],
    activated: &[f32],
    router_norm_weight: &[f32],
    router_weight: &[f32],
    correction_weight: &[f32],
    correct_scale: &[f32],
    rows: usize,
    hidden: usize,
    plane_count: usize,
    active_index: usize,
    eps: f32,
    clip: Option<f32>,
) -> Result<Vec<f32>, String> {
    if predicted_stacked.len() != plane_count * rows * hidden
        || active_prediction.len() != rows * hidden
        || activated.len() != rows * hidden
        || router_norm_weight.len() != hidden
        || router_weight.len() != plane_count * hidden
        || correction_weight.len() != plane_count * plane_count
        || correct_scale.len() != hidden
        || active_index >= plane_count
    {
        return Err("AltUp correction inputs do not match the requested shape".to_string());
    }
    let mut builder = super::builder::Builder::new().with_gemma3n_qmv(true);
    let predicted_arg =
        super::builder::Builder::arg(0, super::builder::Ty::f32(vec![plane_count, rows, hidden]));
    let active_arg = super::builder::Builder::arg(1, super::builder::Ty::f32(vec![rows, hidden]));
    let activated_arg =
        super::builder::Builder::arg(2, super::builder::Ty::f32(vec![rows, hidden]));
    let router_norm_arg = super::builder::Builder::arg(3, super::builder::Ty::f32(vec![hidden]));
    let router_weight_arg =
        super::builder::Builder::arg(4, super::builder::Ty::f32(vec![plane_count, hidden]));
    let correction_arg =
        super::builder::Builder::arg(5, super::builder::Ty::f32(vec![plane_count, plane_count]));
    let scale_arg = super::builder::Builder::arg(6, super::builder::Ty::f32(vec![hidden]));
    let zero = builder.const_f32(0.0);
    let one = builder.const_f32(1.0);
    let eps_value = builder.const_f32(eps);
    let router_norm = super::gemma3n_emit_ops::rms_last_bf16(
        &mut builder,
        &activated_arg,
        Some(&router_norm_arg),
        &eps_value,
        &zero,
    );
    let hidden_value = builder.const_f32(hidden as f32);
    let hidden_value = builder.broadcast(&hidden_value, &[], vec![rows, hidden]);
    let router_scaled = builder.divide(&router_norm, &hidden_value);
    let router_scaled = super::gemma3n_emit_ops::round_bf16(&mut builder, &router_scaled);
    let modalities = builder.gemma3n_qmv(&router_scaled, &router_weight_arg);
    let modalities = builder.gemma3n_tanh(&modalities);
    let correction = match clip {
        Some(limit) => {
            let upper = builder.const_f32(limit);
            let upper = builder.broadcast(&upper, &[], vec![plane_count, plane_count]);
            let lower = builder.const_f32(-limit);
            let lower = builder.broadcast(&lower, &[], vec![plane_count, plane_count]);
            let above = builder.compare("GT", &correction_arg, &upper, "FLOAT");
            let clipped = builder.select(&above, &upper, &correction_arg);
            let below = builder.compare("LT", &clipped, &lower, "FLOAT");
            builder.select(&below, &lower, &clipped)
        }
        None => correction_arg.clone(),
    };
    let coefficients = builder.gemma3n_altup_coeff(&modalities, &correction);
    let one_b = builder.broadcast(&one, &[], vec![rows, plane_count]);
    let coefficients = builder.add(&coefficients, &one_b);
    let innovation = builder.subtract(&activated_arg, &active_arg);
    let innovation = super::gemma3n_emit_ops::round_bf16(&mut builder, &innovation);
    let mut corrections = Vec::with_capacity(plane_count);
    let mut corrected = Vec::with_capacity(plane_count);
    for plane in 0..plane_count {
        let coefficient = builder.slice(&coefficients, &[(0, rows), (plane, plane + 1)]);
        let coefficient = builder.broadcast(&coefficient, &[0, 1], vec![rows, hidden]);
        let predicted = builder.slice(
            &predicted_arg,
            &[(plane, plane + 1), (0, rows), (0, hidden)],
        );
        let predicted = builder.reshape(&predicted, vec![rows, hidden]);
        let (correction, corrected_plane) = super::gemma3n_emit_ops::altup_correct_plane(
            &mut builder,
            &predicted,
            &innovation,
            &coefficient,
        );
        corrections.push(correction.clone());
        corrected.push(corrected_plane);
    }
    let stack = |builder: &mut super::builder::Builder,
                 planes: &[super::builder::Val]|
     -> super::builder::Val {
        let mut stacked = builder.reshape(&planes[0], vec![1, rows, hidden]);
        for plane in &planes[1..] {
            let plane = builder.reshape(plane, vec![1, rows, hidden]);
            stacked = builder.concatenate(&stacked, &plane, 0);
        }
        stacked
    };
    let corrections = stack(&mut builder, &corrections);
    let corrected_stacked = stack(&mut builder, &corrected);
    let corrected_active = corrected[active_index].clone();
    let scale = builder.broadcast(&scale_arg, &[1], vec![rows, hidden]);
    let scaled_active = builder.multiply(&corrected_active, &scale);
    let scaled_active = super::gemma3n_emit_ops::round_bf16(&mut builder, &scaled_active);
    let output = flatten_and_concatenate(
        &mut builder,
        &[
            router_norm,
            router_scaled,
            modalities,
            coefficients,
            innovation,
            corrections,
            corrected_stacked,
            corrected_active,
            scaled_active,
        ],
    );
    let mlir = format!(
        "{target}module @gemma3n_altup_correct_probe {{\n{source}  \
         func.func public @main(%arg0: {predicted_ty}, %arg1: {active_ty}, \
         %arg2: {activated_ty}, %arg3: {router_norm_ty}, \
         %arg4: {router_weight_ty}, %arg5: {correction_ty}, %arg6: {scale_ty}) -> \
         {result_ty} {{\n{body}    return {result} : {result_ty}\n  }}\n}}\n",
        target = target_alias(),
        source = executable_source(),
        predicted_ty = predicted_arg.ty.render(),
        active_ty = active_arg.ty.render(),
        activated_ty = activated_arg.ty.render(),
        router_norm_ty = router_norm_arg.ty.render(),
        router_weight_ty = router_weight_arg.ty.render(),
        correction_ty = correction_arg.ty.render(),
        scale_ty = scale_arg.ty.render(),
        result_ty = output.ty.render(),
        body = builder.body(),
        result = output.name,
    );
    run_cuda_stablehlo_probe(
        "altup-correct",
        mlir,
        &[
            (predicted_stacked, &[plane_count, rows, hidden]),
            (active_prediction, &[rows, hidden]),
            (activated, &[rows, hidden]),
            (router_norm_weight, &[hidden]),
            (router_weight, &[plane_count, hidden]),
            (correction_weight, &[plane_count, plane_count]),
            (correct_scale, &[hidden]),
        ],
        rows * hidden * 5 + rows * plane_count * 2 + plane_count * rows * hidden * 2,
    )
}

#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
fn flatten_and_concatenate(
    builder: &mut super::builder::Builder,
    values: &[super::builder::Val],
) -> super::builder::Val {
    let mut flattened = builder.reshape(
        &values[0],
        vec![values[0].ty.shape.iter().product::<usize>()],
    );
    for value in &values[1..] {
        let value = builder.reshape(value, vec![value.ty.shape.iter().product::<usize>()]);
        flattened = builder.concatenate(&flattened, &value, 0);
    }
    flattened
}

/// Return raw/scaled/RMS/add/inv-sqrt2 PLE stages as one flat diagnostic vector.
#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
pub fn run_ple_diagnostic_probe(
    raw_projection: &[f32],
    token_ple: &[f32],
    norm_weight: &[f32],
    rows: usize,
    layers: usize,
    ple_width: usize,
    hidden: usize,
    eps: f32,
) -> Result<Vec<f32>, String> {
    let total = layers * ple_width;
    if raw_projection.len() != rows * total
        || token_ple.len() != rows * total
        || norm_weight.len() != ple_width
    {
        return Err("PLE diagnostic input lengths do not match the requested shape".to_string());
    }
    let mut builder = super::builder::Builder::new();
    let raw = super::builder::Builder::arg(0, super::builder::Ty::f32(vec![rows, total]));
    let token =
        super::builder::Builder::arg(1, super::builder::Ty::f32(vec![rows, layers, ple_width]));
    let weight = super::builder::Builder::arg(2, super::builder::Ty::f32(vec![ple_width]));
    let scale = super::gemma3n_emit_ops::bf16_scalar(&mut builder, (hidden as f32).sqrt().recip());
    let scale = builder.broadcast(&scale, &[], vec![rows, total]);
    let scaled = builder.multiply(&raw, &scale);
    let scaled = super::gemma3n_emit_ops::round_bf16(&mut builder, &scaled);
    let reshaped = builder.reshape(&scaled, vec![rows, layers, ple_width]);
    let eps = builder.const_f32(eps);
    let zero = builder.const_f32(0.0);
    let normed =
        super::gemma3n_emit_ops::rms_last_bf16(&mut builder, &reshaped, Some(&weight), &eps, &zero);
    let added = builder.add(&normed, &token);
    let added = super::gemma3n_emit_ops::round_bf16(&mut builder, &added);
    let inv = super::gemma3n_emit_ops::bf16_scalar(&mut builder, std::f32::consts::FRAC_1_SQRT_2);
    let inv = builder.broadcast(&inv, &[], vec![rows, layers, ple_width]);
    let combined = builder.multiply(&added, &inv);
    let combined = super::gemma3n_emit_ops::round_bf16(&mut builder, &combined);
    let output = flatten_and_concatenate(
        &mut builder,
        &[raw.clone(), scaled, normed, added, combined],
    );
    let mlir = format!(
        "module @gemma3n_ple_probe {{\n  \
         func.func public @main(%arg0: {raw_ty}, %arg1: {token_ty}, %arg2: {weight_ty}) -> \
         {result_ty} {{\n{body}    return {result} : {result_ty}\n  }}\n}}\n",
        raw_ty = raw.ty.render(),
        token_ty = token.ty.render(),
        weight_ty = weight.ty.render(),
        result_ty = output.ty.render(),
        body = builder.body(),
        result = output.name,
    );
    run_cuda_stablehlo_probe(
        "ple-stages",
        mlir,
        &[
            (raw_projection, &[rows, total]),
            (token_ple, &[rows, layers, ple_width]),
            (norm_weight, &[ple_width]),
        ],
        rows * total * 5,
    )
}

#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
fn bf16_magnitude(
    builder: &mut super::builder::Builder,
    value: &super::builder::Val,
    rows: usize,
    width: usize,
    zero: &super::builder::Val,
) -> super::builder::Val {
    let squared = builder.multiply(value, value);
    let squared = super::gemma3n_emit_ops::round_bf16(builder, &squared);
    let sum = builder.reduce_add(&squared, 1, zero);
    let sum = super::gemma3n_emit_ops::round_bf16(builder, &sum);
    let width = builder.const_f32(width as f32);
    let width = builder.broadcast(&width, &[], vec![rows]);
    let mean = builder.divide(&sum, &width);
    let mean = super::gemma3n_emit_ops::round_bf16(builder, &mean);
    let magnitude = builder.sqrt(&mean);
    super::gemma3n_emit_ops::round_bf16(builder, &magnitude)
}

/// Return target magnitude and all normalized non-base AltUp planes.
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
pub fn run_initial_altup_diagnostic_probe(
    base: &[f32],
    projections: &[f32],
    projection_count: usize,
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Result<Vec<f32>, String> {
    if base.len() != rows * hidden || projections.len() != projection_count * rows * hidden {
        return Err(
            "initial AltUp diagnostic input lengths do not match the requested shape".to_string(),
        );
    }
    let mut builder = super::builder::Builder::new();
    let base_arg = super::builder::Builder::arg(0, super::builder::Ty::f32(vec![rows, hidden]));
    let projections_arg = super::builder::Builder::arg(
        1,
        super::builder::Ty::f32(vec![projection_count, rows, hidden]),
    );
    let zero = builder.const_f32(0.0);
    let target = bf16_magnitude(&mut builder, &base_arg, rows, hidden, &zero);
    let mut outputs = vec![target.clone()];
    for index in 0..projection_count {
        let plane = builder.slice(
            &projections_arg,
            &[(index, index + 1), (0, rows), (0, hidden)],
        );
        let plane = builder.reshape(&plane, vec![rows, hidden]);
        let magnitude = bf16_magnitude(&mut builder, &plane, rows, hidden, &zero);
        let eps_value = builder.const_f32(eps);
        let eps_value = builder.broadcast(&eps_value, &[], vec![rows]);
        let safe_pred = builder.compare("GT", &magnitude, &eps_value, "FLOAT");
        let safe = builder.select(&safe_pred, &magnitude, &eps_value);
        let safe = super::gemma3n_emit_ops::round_bf16(&mut builder, &safe);
        let scale = builder.divide(&target, &safe);
        let scale = super::gemma3n_emit_ops::round_bf16(&mut builder, &scale);
        let scale = builder.broadcast(&scale, &[0], vec![rows, hidden]);
        let normalized = builder.multiply(&plane, &scale);
        outputs.push(super::gemma3n_emit_ops::round_bf16(
            &mut builder,
            &normalized,
        ));
    }
    let output = flatten_and_concatenate(&mut builder, &outputs);
    let mlir = format!(
        "module @gemma3n_initial_altup_probe {{\n  \
         func.func public @main(%arg0: {base_ty}, %arg1: {projections_ty}) -> {result_ty} \
         {{\n{body}    return {result} : {result_ty}\n  }}\n}}\n",
        base_ty = base_arg.ty.render(),
        projections_ty = projections_arg.ty.render(),
        result_ty = output.ty.render(),
        body = builder.body(),
        result = output.name,
    );
    run_cuda_stablehlo_probe(
        "initial-altup-stages",
        mlir,
        &[
            (base, &[rows, hidden]),
            (projections, &[projection_count, rows, hidden]),
        ],
        rows + projection_count * rows * hidden,
    )
}

/// Return router-norm/scaled/modalities/predicted/active/input-norm stages.
#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
pub fn run_altup_predict_diagnostic_probe(
    planes: &[f32],
    router_norm_weight: &[f32],
    router_weight: &[f32],
    prediction_weight: &[f32],
    input_norm_weight: &[f32],
    plane_count: usize,
    active_index: usize,
    rows: usize,
    hidden: usize,
    eps: f32,
    clip: Option<f32>,
) -> Result<Vec<f32>, String> {
    if planes.len() != plane_count * rows * hidden
        || router_norm_weight.len() != hidden
        || router_weight.len() != plane_count * hidden
        || prediction_weight.len() != plane_count * plane_count * plane_count
        || input_norm_weight.len() != hidden
        || active_index >= plane_count
    {
        return Err("AltUp predict diagnostic input lengths do not match the shape".to_string());
    }
    let mut builder = super::builder::Builder::new().with_gemma3n_qmv(true);
    let planes_arg =
        super::builder::Builder::arg(0, super::builder::Ty::f32(vec![plane_count, rows, hidden]));
    let router_norm_arg = super::builder::Builder::arg(1, super::builder::Ty::f32(vec![hidden]));
    let router_weight_arg =
        super::builder::Builder::arg(2, super::builder::Ty::f32(vec![plane_count, hidden]));
    let prediction_arg = super::builder::Builder::arg(
        3,
        super::builder::Ty::f32(vec![plane_count * plane_count, plane_count]),
    );
    let input_norm_arg = super::builder::Builder::arg(4, super::builder::Ty::f32(vec![hidden]));
    let active = builder.slice(
        &planes_arg,
        &[(active_index, active_index + 1), (0, rows), (0, hidden)],
    );
    let active = builder.reshape(&active, vec![rows, hidden]);
    let eps_value = builder.const_f32(eps);
    let zero = builder.const_f32(0.0);
    let routed_norm = super::gemma3n_emit_ops::rms_last_bf16(
        &mut builder,
        &active,
        Some(&router_norm_arg),
        &eps_value,
        &zero,
    );
    let hidden_value = builder.const_f32(hidden as f32);
    let hidden_value = builder.broadcast(&hidden_value, &[], vec![rows, hidden]);
    let routed_scaled = builder.divide(&routed_norm, &hidden_value);
    let routed_scaled = super::gemma3n_emit_ops::round_bf16(&mut builder, &routed_scaled);
    let modalities = builder.gemma3n_qmv(&routed_scaled, &router_weight_arg);
    let modalities = builder.gemma3n_tanh(&modalities);
    let prediction = match clip {
        Some(limit) => {
            let positive = builder.const_f32(limit);
            let positive =
                builder.broadcast(&positive, &[], vec![plane_count * plane_count, plane_count]);
            let negative = builder.const_f32(-limit);
            let negative =
                builder.broadcast(&negative, &[], vec![plane_count * plane_count, plane_count]);
            let above = builder.compare("GT", &prediction_arg, &positive, "FLOAT");
            let clipped = builder.select(&above, &positive, &prediction_arg);
            let below = builder.compare("LT", &clipped, &negative, "FLOAT");
            builder.select(&below, &negative, &clipped)
        }
        None => prediction_arg.clone(),
    };
    let coefficients = builder.gemma3n_altup_coeff(&modalities, &prediction);
    let coefficients = builder.reshape(&coefficients, vec![rows, plane_count, plane_count]);
    let coefficients = builder.transpose(&coefficients, &[0, 2, 1]);
    let predicted = builder.gemma3n_altup_predict(&planes_arg, &coefficients);
    let active_predicted = builder.slice(
        &predicted,
        &[(active_index, active_index + 1), (0, rows), (0, hidden)],
    );
    let active_predicted = builder.reshape(&active_predicted, vec![rows, hidden]);
    let input_norm = super::gemma3n_emit_ops::rms_last_bf16(
        &mut builder,
        &active_predicted,
        Some(&input_norm_arg),
        &eps_value,
        &zero,
    );
    let output = flatten_and_concatenate(
        &mut builder,
        &[
            routed_norm,
            routed_scaled,
            modalities,
            coefficients,
            predicted,
            active_predicted,
            input_norm,
        ],
    );
    let mlir = format!(
        "{target}module @gemma3n_altup_predict_probe {{\n{source}  \
         func.func public @main(%arg0: {planes_ty}, %arg1: {router_norm_ty}, \
         %arg2: {router_weight_ty}, %arg3: {prediction_ty}, %arg4: {input_norm_ty}) -> \
         {result_ty} {{\n{body}    return {result} : {result_ty}\n  }}\n}}\n",
        target = target_alias(),
        source = executable_source(),
        planes_ty = planes_arg.ty.render(),
        router_norm_ty = router_norm_arg.ty.render(),
        router_weight_ty = router_weight_arg.ty.render(),
        prediction_ty = prediction_arg.ty.render(),
        input_norm_ty = input_norm_arg.ty.render(),
        result_ty = output.ty.render(),
        body = builder.body(),
        result = output.name,
    );
    let output_len = rows * hidden * 2
        + rows * hidden * 2
        + rows * plane_count
        + rows * plane_count * plane_count
        + plane_count * rows * hidden;
    run_cuda_stablehlo_probe(
        "altup-predict-stages",
        mlir,
        &[
            (planes, &[plane_count, rows, hidden]),
            (router_norm_weight, &[hidden]),
            (router_weight, &[plane_count, hidden]),
            (prediction_weight, &[plane_count * plane_count, plane_count]),
            (input_norm_weight, &[hidden]),
        ],
        output_len,
    )
}

#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
#[allow(clippy::too_many_arguments)]
pub fn run_ple_injection_diagnostic_probe(
    corrected_active: &[f32],
    correct_scale: &[f32],
    per_layer_input: &[f32],
    gate_weight: &[f32],
    projection_weight: &[f32],
    norm_weight: &[f32],
    residual_plane: &[f32],
    rows: usize,
    hidden: usize,
    ple_width: usize,
    eps: f32,
) -> Result<Vec<f32>, String> {
    if corrected_active.len() != rows * hidden
        || correct_scale.len() != hidden
        || per_layer_input.len() != rows * ple_width
        || gate_weight.len() != ple_width * hidden
        || projection_weight.len() != hidden * ple_width
        || norm_weight.len() != hidden
        || residual_plane.len() != rows * hidden
    {
        return Err("PLE injection diagnostic input lengths do not match the shape".to_string());
    }

    let mut builder = super::builder::Builder::new().with_gemma3n_qmv(true);
    let active_arg = super::builder::Builder::arg(0, super::builder::Ty::f32(vec![rows, hidden]));
    let scale_arg = super::builder::Builder::arg(1, super::builder::Ty::f32(vec![hidden]));
    let ple_arg = super::builder::Builder::arg(2, super::builder::Ty::f32(vec![rows, ple_width]));
    let gate_arg =
        super::builder::Builder::arg(3, super::builder::Ty::f32(vec![ple_width, hidden]));
    let projection_arg =
        super::builder::Builder::arg(4, super::builder::Ty::f32(vec![hidden, ple_width]));
    let norm_arg = super::builder::Builder::arg(5, super::builder::Ty::f32(vec![hidden]));
    let residual_arg = super::builder::Builder::arg(6, super::builder::Ty::f32(vec![rows, hidden]));

    let scale = builder.broadcast(&scale_arg, &[1], vec![rows, hidden]);
    let scaled_active = builder.multiply(&active_arg, &scale);
    let scaled_active = super::gemma3n_emit_ops::round_bf16(&mut builder, &scaled_active);
    let gate = builder.gemma3n_qmv(&scaled_active, &gate_arg);
    let activated = super::gemma3n_emit_ops::geglu(&mut builder, &gate, &ple_arg);
    let projected = builder.gemma3n_qmv(&activated, &projection_arg);
    let eps_value = builder.const_f32(eps);
    let zero = builder.const_f32(0.0);
    let injected = super::gemma3n_emit_ops::rms_last_bf16(
        &mut builder,
        &projected,
        Some(&norm_arg),
        &eps_value,
        &zero,
    );
    let updated = builder.add(&residual_arg, &injected);
    let updated = super::gemma3n_emit_ops::round_bf16(&mut builder, &updated);
    let outputs = vec![scaled_active, gate, activated, projected, injected, updated];
    let output = flatten_and_concatenate(&mut builder, &outputs);
    let mlir = format!(
        "{target}module @gemma3n_ple_injection_probe {{\n{source}  \
         func.func public @main(%arg0: {active_ty}, %arg1: {scale_ty}, \
         %arg2: {ple_ty}, %arg3: {gate_ty}, %arg4: {projection_ty}, \
         %arg5: {norm_ty}, %arg6: {residual_ty}) -> {result_ty} {{\n\
         {body}    return {result} : {result_ty}\n  }}\n}}\n",
        target = target_alias(),
        source = executable_source(),
        active_ty = active_arg.ty.render(),
        scale_ty = scale_arg.ty.render(),
        ple_ty = ple_arg.ty.render(),
        gate_ty = gate_arg.ty.render(),
        projection_ty = projection_arg.ty.render(),
        norm_ty = norm_arg.ty.render(),
        residual_ty = residual_arg.ty.render(),
        result_ty = output.ty.render(),
        body = builder.body(),
        result = output.name,
    );
    let output_len = rows * (hidden * 4 + ple_width * 2);
    run_cuda_stablehlo_probe(
        "ple-injection-stages",
        mlir,
        &[
            (corrected_active, &[rows, hidden]),
            (correct_scale, &[hidden]),
            (per_layer_input, &[rows, ple_width]),
            (gate_weight, &[ple_width, hidden]),
            (projection_weight, &[hidden, ple_width]),
            (norm_weight, &[hidden]),
            (residual_plane, &[rows, hidden]),
        ],
        output_len,
    )
}

/// Isolate the PLE injection and every non-active AltUp residual update using
/// exact MLX carriers from one logical layer.
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
#[allow(clippy::too_many_arguments)]
pub fn run_ple_injection_all_planes_diagnostic_probe(
    corrected_active: &[f32],
    correct_scale: &[f32],
    per_layer_input: &[f32],
    gate_weight: &[f32],
    projection_weight: &[f32],
    norm_weight: &[f32],
    residual_planes: &[f32],
    rows: usize,
    hidden: usize,
    ple_width: usize,
    non_active_planes: usize,
    eps: f32,
) -> Result<Vec<f32>, String> {
    if non_active_planes == 0
        || corrected_active.len() != rows * hidden
        || correct_scale.len() != hidden
        || per_layer_input.len() != rows * ple_width
        || gate_weight.len() != ple_width * hidden
        || projection_weight.len() != hidden * ple_width
        || norm_weight.len() != hidden
        || residual_planes.len() != non_active_planes * rows * hidden
    {
        return Err("all-plane PLE diagnostic inputs do not match the shape".to_string());
    }

    let mut builder = super::builder::Builder::new().with_gemma3n_qmv(true);
    let active_arg = super::builder::Builder::arg(0, super::builder::Ty::f32(vec![rows, hidden]));
    let scale_arg = super::builder::Builder::arg(1, super::builder::Ty::f32(vec![hidden]));
    let ple_arg = super::builder::Builder::arg(2, super::builder::Ty::f32(vec![rows, ple_width]));
    let gate_arg =
        super::builder::Builder::arg(3, super::builder::Ty::f32(vec![ple_width, hidden]));
    let projection_arg =
        super::builder::Builder::arg(4, super::builder::Ty::f32(vec![hidden, ple_width]));
    let norm_arg = super::builder::Builder::arg(5, super::builder::Ty::f32(vec![hidden]));
    let residual_arg = super::builder::Builder::arg(
        6,
        super::builder::Ty::f32(vec![non_active_planes, rows, hidden]),
    );

    let scale = builder.broadcast(&scale_arg, &[1], vec![rows, hidden]);
    let scaled_active = builder.multiply(&active_arg, &scale);
    let scaled_active = super::gemma3n_emit_ops::round_bf16(&mut builder, &scaled_active);
    let gate = builder.gemma3n_qmv(&scaled_active, &gate_arg);
    let activated = super::gemma3n_emit_ops::geglu(&mut builder, &gate, &ple_arg);
    let projected = builder.gemma3n_qmv(&activated, &projection_arg);
    let eps_value = builder.const_f32(eps);
    let zero = builder.const_f32(0.0);
    let injected = super::gemma3n_emit_ops::rms_last_bf16(
        &mut builder,
        &projected,
        Some(&norm_arg),
        &eps_value,
        &zero,
    );
    let injected_planes =
        builder.broadcast(&injected, &[1, 2], vec![non_active_planes, rows, hidden]);
    let residual_sums = builder.add(&residual_arg, &injected_planes);
    let residuals_updated = super::gemma3n_emit_ops::round_bf16(&mut builder, &residual_sums);
    let outputs = vec![
        scaled_active,
        gate,
        activated,
        projected,
        injected,
        residual_arg.clone(),
        residual_sums,
        residuals_updated,
    ];
    let output = flatten_and_concatenate(&mut builder, &outputs);
    let mlir = format!(
        "{target}module @gemma3n_ple_injection_all_planes_probe {{\n{source}  \
         func.func public @main(%arg0: {active_ty}, %arg1: {scale_ty}, \
         %arg2: {ple_ty}, %arg3: {gate_ty}, %arg4: {projection_ty}, \
         %arg5: {norm_ty}, %arg6: {residual_ty}) -> {result_ty} {{\n\
         {body}    return {result} : {result_ty}\n  }}\n}}\n",
        target = target_alias(),
        source = executable_source(),
        active_ty = active_arg.ty.render(),
        scale_ty = scale_arg.ty.render(),
        ple_ty = ple_arg.ty.render(),
        gate_ty = gate_arg.ty.render(),
        projection_ty = projection_arg.ty.render(),
        norm_ty = norm_arg.ty.render(),
        residual_ty = residual_arg.ty.render(),
        result_ty = output.ty.render(),
        body = builder.body(),
        result = output.name,
    );
    let output_len = rows * (hidden * 3 + ple_width * 2) + non_active_planes * rows * hidden * 3;
    run_cuda_stablehlo_probe(
        "ple-injection-all-planes-stages",
        mlir,
        &[
            (corrected_active, &[rows, hidden]),
            (correct_scale, &[hidden]),
            (per_layer_input, &[rows, ple_width]),
            (gate_weight, &[ple_width, hidden]),
            (projection_weight, &[hidden, ple_width]),
            (norm_weight, &[hidden]),
            (residual_planes, &[non_active_planes, rows, hidden]),
        ],
        output_len,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(xla_iree_cuda)]
    use crate::emitter::builder::Builder;
    use crate::weights::round_bf16_f32;
    #[cfg(xla_iree_cuda)]
    use std::path::PathBuf;
    #[cfg(xla_iree_cuda)]
    use std::process::Command;

    fn reference_qmv(input: &[f32], weight: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        assert_eq!(input.len(), m * k);
        assert_eq!(weight.len(), n * k);
        let mut output = vec![0.0; m * n];
        for row in 0..m {
            for output_row in 0..n {
                let mut lane_sums = [0.0f32; 32];
                for (lane, lane_sum) in lane_sums.iter_mut().enumerate() {
                    let mut slots = [0.0f32; 16];
                    for round in 0..k.div_ceil(512) {
                        for (i, slot) in slots.iter_mut().enumerate() {
                            let column = lane * 16 + i + round * 512;
                            if column < k {
                                let lhs = round_bf16_f32(input[row * k + column]);
                                let rhs = round_bf16_f32(weight[output_row * k + column]);
                                *slot = round_bf16_f32(lhs.mul_add(rhs, *slot));
                            }
                        }
                    }
                    let mut sum = 0.0f32;
                    for slot in slots {
                        sum += slot;
                    }
                    *lane_sum = sum;
                }
                for mask in [16, 8, 4, 2, 1] {
                    let previous = lane_sums;
                    for lane in 0..32 {
                        lane_sums[lane] += previous[lane ^ mask];
                    }
                }
                output[row * n + output_row] = round_bf16_f32(lane_sums[0]);
            }
        }
        output
    }

    fn synthetic_q4_carriers(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
        let input = (0..m * k)
            .map(|index| {
                let signed = ((index * 17 + index / 7) % 101) as i32 - 50;
                round_bf16_f32(signed as f32 * 0.03125)
            })
            .collect();
        let weight = (0..n * k)
            .map(|index| {
                let signed = ((index * 29 + index / 11) % 127) as i32 - 63;
                round_bf16_f32(signed as f32 * 0.015625)
            })
            .collect();
        (input, weight)
    }

    fn round_tf32_rne_bits(bits: u32) -> u32 {
        if bits & 0x7f80_0000 == 0x7f80_0000 {
            return bits;
        }
        let retained_lsb = (bits >> 13) & 1;
        bits.wrapping_add(0x0fff + retained_lsb) & 0xffff_e000
    }

    #[cfg(xla_iree_cuda)]
    fn probe_mlir(m: usize, n: usize, k: usize) -> String {
        let mut builder = Builder::new().with_gemma3n_qmv(true);
        let input = Builder::arg(0, crate::emitter::builder::Ty::f32(vec![m, k]));
        let weight = Builder::arg(1, crate::emitter::builder::Ty::f32(vec![n, k]));
        let output = builder.gemma3n_qmv(&input, &weight);
        format!(
            "{target}module @gemma3n_qmv_probe {{\n{source}  \
             func.func public @main(%arg0: {input_ty}, %arg1: {weight_ty}) -> {result_ty} \
             {{\n{body}    \
             return {result} : {result_ty}\n  }}\n}}\n",
            target = target_alias(),
            source = executable_source(),
            input_ty = input.ty.render(),
            weight_ty = weight.ty.render(),
            result_ty = output.ty.render(),
            body = builder.body(),
            result = output.name,
        )
    }

    #[cfg(xla_iree_cuda)]
    fn sdpa_dynamic_position_probe_mlir() -> String {
        const CAPACITY: usize = 8;
        const QUERY_HEADS: usize = 8;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 256;
        let mut builder = Builder::new().with_gemma3n_qmv(true);
        let query = Builder::arg(
            0,
            crate::emitter::builder::Ty::f32(vec![QUERY_HEADS, HEAD_DIM]),
        );
        let keys = Builder::arg(
            1,
            crate::emitter::builder::Ty::f32(vec![CAPACITY, KV_HEADS, HEAD_DIM]),
        );
        let values = Builder::arg(
            2,
            crate::emitter::builder::Ty::f32(vec![CAPACITY, KV_HEADS, HEAD_DIM]),
        );
        let position = Builder::arg(3, crate::emitter::builder::Ty::scalar("i32"));
        let output = builder.gemma3n_sdpa_vector(&query, &keys, &values, &position, None, 1.0);
        format!(
            "{target}module @gemma3n_sdpa_dynamic_position_probe {{\n{source}  \
             func.func public @main(%arg0: {query_ty}, %arg1: {keys_ty}, \
             %arg2: {values_ty}, %arg3: tensor<i32>) -> {result_ty} {{\n{body}    \
             return {result} : {result_ty}\n  }}\n}}\n",
            target = target_alias(),
            source = executable_source(),
            query_ty = query.ty.render(),
            keys_ty = keys.ty.render(),
            values_ty = values.ty.render(),
            result_ty = output.ty.render(),
            body = builder.body(),
            result = output.name,
        )
    }

    #[cfg(xla_iree_cuda)]
    fn required_path(env_name: &str, fallback: impl FnOnce() -> Option<PathBuf>) -> PathBuf {
        std::env::var_os(env_name)
            .map(PathBuf::from)
            .or_else(fallback)
            .unwrap_or_else(|| panic!("set {env_name} for the CUDA QMV parity test"))
    }

    #[test]
    fn metadata_identity_pins_numerical_abi() {
        let identity = artifact_identity();
        assert!(identity.contains("kernel-v8"));
        assert!(identity.contains("abi-v6"));
        assert!(identity.contains("min-sm80"));
        assert!(identity.contains("ptx-fnv1a64="));
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn sdpa_production_abi_passes_dynamic_position_and_exact_scale_bits() {
        let mut builder = Builder::new().with_gemma3n_qmv(true);
        let query = Builder::arg(0, crate::emitter::builder::Ty::f32(vec![8, 256]));
        let keys = Builder::arg(1, crate::emitter::builder::Ty::f32(vec![8, 2, 256]));
        let values = Builder::arg(2, crate::emitter::builder::Ty::f32(vec![8, 2, 256]));
        let position = Builder::arg(3, crate::emitter::builder::Ty::scalar("i32"));
        let _ = builder.gemma3n_sdpa_vector(&query, &keys, &values, &position, Some(512), 1.0);
        let body = builder.body();
        assert!(body.contains("arith.constant 1065353216 : i32"));
        assert!(body.contains("(i32, i32, i32, i32, i32, i32,"));
        assert!(body.contains("tensor.extract %arg3[] : tensor<i32>"));
        assert!(!body.contains("arith.constant 1.000000000 : f32"));
        let source = executable_source();
        assert!(source.contains("@gemma3n_sdpa_vector ordinal(5)"));
        assert!(source.contains("#hal.pipeline.layout<constants = 6"));

        let invalid_gqa = std::panic::catch_unwind(|| {
            let mut builder = Builder::new().with_gemma3n_qmv(true);
            let query = Builder::arg(0, crate::emitter::builder::Ty::f32(vec![3, 256]));
            let keys = Builder::arg(1, crate::emitter::builder::Ty::f32(vec![8, 2, 256]));
            let values = Builder::arg(2, crate::emitter::builder::Ty::f32(vec![8, 2, 256]));
            let position = Builder::arg(3, crate::emitter::builder::Ty::scalar("i32"));
            builder.gemma3n_sdpa_vector(&query, &keys, &values, &position, None, 1.0)
        });
        assert!(invalid_gqa.is_err());

        let invalid_dimension = std::panic::catch_unwind(|| {
            let mut builder = Builder::new().with_gemma3n_qmv(true);
            let query = Builder::arg(0, crate::emitter::builder::Ty::f32(vec![8, 128]));
            let keys = Builder::arg(1, crate::emitter::builder::Ty::f32(vec![8, 2, 128]));
            let values = Builder::arg(2, crate::emitter::builder::Ty::f32(vec![8, 2, 128]));
            let position = Builder::arg(3, crate::emitter::builder::Ty::scalar("i32"));
            builder.gemma3n_sdpa_vector(&query, &keys, &values, &position, None, 1.0)
        });
        assert!(invalid_dimension.is_err());

        let invalid_capacity = std::panic::catch_unwind(|| {
            let mut builder = Builder::new().with_gemma3n_qmv(true);
            let query = Builder::arg(0, crate::emitter::builder::Ty::f32(vec![8, 256]));
            let keys = Builder::arg(1, crate::emitter::builder::Ty::f32(vec![1025, 2, 256]));
            let values = Builder::arg(2, crate::emitter::builder::Ty::f32(vec![1025, 2, 256]));
            let position = Builder::arg(3, crate::emitter::builder::Ty::scalar("i32"));
            builder.gemma3n_sdpa_vector(&query, &keys, &values, &position, None, 1.0)
        });
        assert!(invalid_capacity.is_err());

        let invalid_window = std::panic::catch_unwind(|| {
            let mut builder = Builder::new().with_gemma3n_qmv(true);
            let query = Builder::arg(0, crate::emitter::builder::Ty::f32(vec![8, 256]));
            let keys = Builder::arg(1, crate::emitter::builder::Ty::f32(vec![64, 2, 256]));
            let values = Builder::arg(2, crate::emitter::builder::Ty::f32(vec![64, 2, 256]));
            let position = Builder::arg(3, crate::emitter::builder::Ty::scalar("i32"));
            builder.gemma3n_sdpa_vector(&query, &keys, &values, &position, Some(8), 1.0)
        });
        assert!(invalid_window.is_err());

        let cuda = include_str!("../../csrc/gemma3n_qmv.cu");
        assert!(cuda.contains("live > 1024"));
        assert!(cuda.contains("query_heads % kv_heads != 0"));
        assert!(cuda.contains("window != 0 && window < live"));
    }

    #[cfg(xla_iree_cuda)]
    #[test]
    #[ignore = "requires the pinned CUDA-capable IREE compiler"]
    fn sdpa_dynamic_tensor_position_compiles_to_scalar_push_constant() {
        let compiler = required_path("MLXCEL_XLA_IREE_COMPILE", || {
            option_env!("MLXCEL_XLA_IREE_COMPILE").map(PathBuf::from)
        });
        let stem = format!("mlxcel-gemma3n-sdpa-position-{}", std::process::id());
        let mlir_path = std::env::temp_dir().join(format!("{stem}.mlir"));
        let vmfb_path = std::env::temp_dir().join(format!("{stem}.vmfb"));
        std::fs::write(&mlir_path, sdpa_dynamic_position_probe_mlir()).unwrap();
        let compiled = Command::new(&compiler)
            .arg("--iree-input-type=stablehlo")
            .arg("--iree-hal-target-device=cuda")
            .arg("--iree-cuda-target=sm_80")
            .arg(&mlir_path)
            .arg("-o")
            .arg(&vmfb_path)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&mlir_path);
        let _ = std::fs::remove_file(&vmfb_path);
        assert!(
            compiled.status.success(),
            "dynamic-position SDPA failed typed IREE compile:\n{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
    }

    #[test]
    fn tf32_halfway_rounds_to_nearest_even() {
        const HALFWAY_EVEN: u32 = 0xbe5d_5000;
        assert_eq!(round_tf32_rne_bits(HALFWAY_EVEN), 0xbe5d_4000);
        assert_ne!(
            round_tf32_rne_bits(HALFWAY_EVEN),
            HALFWAY_EVEN.wrapping_add(0x1000) & 0xffff_e000,
        );
    }

    #[test]
    fn native_altup_source_pins_rne_and_tensor_core_schedule() {
        let source = include_str!("../../csrc/gemma3n_qmv.cu");
        assert!(source.contains("float gemma3n_to_tf32_rne(float value)"));
        assert!(source.contains("0x0FFFu + retained_lsb"));
        assert!(!source.contains("cvt.rna.tf32.f32 converted"));
        assert_eq!(
            source
                .matches("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32")
                .count(),
            1,
        );
        assert!(!source.contains("void gemma3n_altup_predict_mma("));
    }

    #[cfg(xla_iree_cuda)]
    #[test]
    fn native_altup_ptx_has_one_mma_and_no_scalar_reduction() {
        let ptx = std::str::from_utf8(ptx()).expect("nvcc PTX is UTF-8 text");
        let entry = ptx
            .find(".visible .entry gemma3n_altup_predict(")
            .expect("production AltUp prediction entry");
        let tail = &ptx[entry..];
        let end = tail[1..]
            .find(".visible .entry ")
            .map_or(tail.len(), |offset| offset + 1);
        let body = &tail[..end];
        assert_eq!(
            body.matches("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32")
                .count(),
            1,
        );
        assert!(!body.contains("fma.rn.f32"));
        assert!(!body.contains("fma.rn.ftz.f32"));

        let source = executable_source();
        assert!(source.contains("(s0 ceildiv 16)>()[%hidden]"));
        assert!(source.contains("hal.return %x, %rows, %c1"));
        assert!(source.contains("workgroup_size = [32 : index, 1 : index, 1 : index]"));
    }

    #[test]
    fn native_geglu_pins_mlx_bf16_fma_boundary() {
        let source = include_str!("../../csrc/gemma3n_qmv.cu");
        assert!(source.contains("void gemma3n_geglu_bf16("));
        assert!(source.contains("__hfma(x3, cubic, x)"));
    }

    #[test]
    fn synthetic_q4_group64_schedule_has_pinned_output_bits() {
        const M: usize = 2;
        const N: usize = 9;
        const K: usize = 576;
        let (input, weight) = synthetic_q4_carriers(M, N, K);
        let bits = reference_qmv(&input, &weight, M, N, K)
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            bits,
            vec![
                3238133760, 3231449088, 3234856960, 1080819712, 1086062592, 1082785792, 1083703296,
                1079902208, 1082916864, 3238920192, 3212836864, 3239378944, 3207856128, 3231121408,
                1077936128, 3225681920, 1090650112, 1066139648,
            ]
        );
    }

    #[cfg(xla_iree_cuda)]
    #[test]
    #[ignore = "requires pinned IREE compiler/runtime plus a CUDA device"]
    fn actual_cuda_dispatch_matches_q4_group64_reference_bit_exactly() {
        const M: usize = 2;
        const N: usize = 9;
        const K: usize = 576;
        let (input, weight) = synthetic_q4_carriers(M, N, K);
        let expected = reference_qmv(&input, &weight, M, N, K);
        let mlir = probe_mlir(M, N, K);

        let compiler = required_path("MLXCEL_XLA_IREE_COMPILE", || {
            option_env!("MLXCEL_XLA_IREE_COMPILE").map(PathBuf::from)
        });
        let runner = required_path("MLXCEL_XLA_IREE_RUN_MODULE", || {
            std::env::var_os("IREE_CUDA_HOME").map(|home| {
                PathBuf::from(home)
                    .join("build")
                    .join("tools")
                    .join("iree-run-module")
            })
        });
        assert!(compiler.is_file(), "missing {}", compiler.display());
        assert!(runner.is_file(), "missing {}", runner.display());

        let stem = format!("mlxcel-gemma3n-qmv-probe-{}", std::process::id());
        let mlir_path = std::env::temp_dir().join(format!("{stem}.mlir"));
        let vmfb_path = std::env::temp_dir().join(format!("{stem}.vmfb"));
        let input_path = std::env::temp_dir().join(format!("{stem}-input.bin"));
        let weight_path = std::env::temp_dir().join(format!("{stem}-weight.bin"));
        let output_path = std::env::temp_dir().join(format!("{stem}.bin"));
        std::fs::write(&mlir_path, mlir).unwrap();
        std::fs::write(
            &input_path,
            input
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        std::fs::write(
            &weight_path,
            weight
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let compile = Command::new(&compiler)
            .arg("--iree-input-type=stablehlo")
            .arg("--iree-hal-target-device=cuda")
            .arg("--iree-cuda-target=sm_80")
            .arg(&mlir_path)
            .arg("-o")
            .arg(&vmfb_path)
            .output()
            .unwrap();
        assert!(
            compile.status.success(),
            "iree-compile failed:\n{}",
            String::from_utf8_lossy(&compile.stderr)
        );
        let run = Command::new(&runner)
            .arg("--device=cuda")
            .arg(format!("--module={}", vmfb_path.display()))
            .arg("--function=main")
            .arg(format!("--input={M}x{K}xf32=@{}", input_path.display()))
            .arg(format!("--input={N}x{K}xf32=@{}", weight_path.display()))
            .arg(format!("--output=@{}", output_path.display()))
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "iree-run-module failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr),
        );
        let bytes = std::fs::read(&output_path).unwrap();
        assert_eq!(bytes.len(), expected.len() * std::mem::size_of::<f32>());
        let actual = bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );

        for path in [mlir_path, vmfb_path, input_path, weight_path, output_path] {
            let _ = std::fs::remove_file(path);
        }
    }

    #[cfg(all(feature = "diagnostics", xla_iree_cuda))]
    #[test]
    #[ignore = "requires pinned IREE compiler/runtime plus a CUDA device"]
    fn actual_cuda_altup_coeff_rounds_tf32_halfway_to_even() {
        let input = [f32::from_bits(0xbe5d_5000)];
        let weight = [1.0f32];
        let mut builder = Builder::new().with_gemma3n_qmv(true);
        let input_arg = Builder::arg(0, crate::emitter::builder::Ty::f32(vec![1, 1]));
        let weight_arg = Builder::arg(1, crate::emitter::builder::Ty::f32(vec![1, 1]));
        let output = builder.gemma3n_altup_coeff(&input_arg, &weight_arg);
        let mlir = format!(
            "{target}module @gemma3n_altup_coeff_rne_probe {{\n{source}  \
             func.func public @main(%arg0: {input_ty}, %arg1: {weight_ty}) -> {result_ty} \
             {{\n{body}    return {result} : {result_ty}\n  }}\n}}\n",
            target = target_alias(),
            source = executable_source(),
            input_ty = input_arg.ty.render(),
            weight_ty = weight_arg.ty.render(),
            result_ty = output.ty.render(),
            body = builder.body(),
            result = output.name,
        );
        let actual = run_cuda_stablehlo_probe(
            "altup-coeff-rne-halfway",
            mlir,
            &[(&input, &[1, 1]), (&weight, &[1, 1])],
            1,
        )
        .unwrap();
        assert_eq!(actual[0].to_bits(), 0xbe5d_4000);
    }
}
