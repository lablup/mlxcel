// ROCm lowering tuning spec (issues #2873, #2921).
//
// Applied to the model graphs on the hip backend. Two contraction shapes get a
// lowering config that the compiler defaults handle poorly on gfx1151; every
// other shape keeps the default config.
//
// 1. Long-K single-row contractions (`match_bigk_vecmat`). Raises the reduction
//    tile from the compiler default [0, 4] to [0, 32]: the down projection
//    (K = intermediate, 8192 on llama-3.2-1b) moves from about 50 GB/s to about
//    110 GB/s in a standalone trace, while the same tile is slower on short-K
//    vecmats, so the match requires K >= 4096. Shape match: a rank-1 by rank-2
//    contraction with f16 operands and an f32 accumulator, no batch dims.
//
// 2. The grouped-query decode score contraction (`match_gqa_score`). It is
//    [batch = n_kv, M = group, K = head_dim] by [batch, K, N = keys] f16 into
//    an f32 result, so M is 4 and K is 64 on llama-3.2-1b. The default config
//    for that shape picks a 32-thread workgroup and a [1, 1, 128] thread tile
//    and runs the dispatch at about 74 us; the structurally similar context
//    matmul (K = 256) runs at about 6.5 us for the same MAC count. The config
//    below tiles all four query rows and 32 keys per workgroup with the whole
//    64-wide reduction in one step and drops the dispatch to about 1.5 us
//    (issue #2921). The match is deliberately narrow: batch = n_kv = 8, M = 4,
//    K = 64 exactly and N >= 128, which is the measured shape of the
//    single-sequence decode graph for context capacities 128 and up. The batch
//    predicate is what keeps the ragged and uniform-B graphs out: their
//    rank-4 dot_general collapses to the same rank-3 contraction with the same
//    maps, but with `bsz * n_kv` as the batch, so a multi-row graph has 16 or
//    more and keeps the compiler default. Other group sizes, head widths and
//    batch counts keep the compiler default until they are measured.
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

  transform.named_sequence @match_gqa_score(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %batch_dims, %m_dims, %n_dims, %k_dims = transform.iree.match.contraction %arg0,
      lhs_type = f16, rhs_type = f16, output_type = f32,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
      : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch_dims, [8] : !transform.param<i64>
    transform.iree.match.dims_equal %m_dims, [4] : !transform.param<i64>
    transform.iree.match.dims_equal %k_dims, [64] : !transform.param<i64>
    %rhs = transform.get_operand %arg0[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %rhs[2], umin = 128, none : !transform.any_value
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{reduction = [0, 0, 0, 64], thread = [1, 1, 1, 0], workgroup = [1, 4, 32, 1]}>, translation_info = <pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [32, 4, 1] subgroup_size = 32>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed}) -> !transform.any_op
      attributes {iree_codegen.tuning_spec_entrypoint} {
    %updated = transform.foreach_match in %arg0
      @match_bigk_vecmat -> @apply_op_config,
      @match_gqa_score -> @apply_op_config
      : (!transform.any_op) -> !transform.any_op
    transform.yield %updated : !transform.any_op
  }
}
