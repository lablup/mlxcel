// ROCm lowering tuning spec (issue #2873).
//
// Applied to the model graphs on the hip backend. It raises the reduction tile
// of long-K single-row contractions from the compiler default [0, 4] to
// [0, 32]: the down projection (K = intermediate, 8192 on llama-3.2-1b) moves
// from about 50 GB/s to about 110 GB/s in a standalone trace, while the same
// tile is slower on short-K vecmats, so the match requires K >= 4096.
//
// Shape match: a rank-1 by rank-2 contraction (vecmat) with f16 operands and an
// f32 accumulator, no batch dims. Every other shape keeps the default config.
module attributes {iree_codegen.tuning_spec_with_default_entrypoint, transform.with_named_sequence} {
  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}, %config: !transform.any_param {transform.readonly}) {
    transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_bigk_vecmat(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %batch_dims, %m_dims, %n_dims, %k_dims = transform.iree.match.contraction %arg0,
      lhs_type = f16, rhs_type = f16, output_type = f32,
      indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>]
      : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch_dims, [] : !transform.param<i64>
    %rhs = transform.get_operand %arg0[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %rhs[0], umin = 4096, none : !transform.any_value
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{reduction = [0, 32], thread = [8, 0], workgroup = [256, 0]}>, translation_info = <pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [32, 1, 1] subgroup_size = 32>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed}) -> !transform.any_op
      attributes {iree_codegen.tuning_spec_entrypoint} {
    %updated = transform.foreach_match in %arg0
      @match_bigk_vecmat -> @apply_op_config : (!transform.any_op) -> !transform.any_op
    transform.yield %updated : !transform.any_op
  }
}
