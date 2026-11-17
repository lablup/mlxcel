// ROCm lowering tuning spec (issues #2873, #2921, #2956).
//
// Applied to the model graphs on the hip backend. Four contraction classes get
// a lowering config that the compiler defaults handle poorly on gfx1151; every
// other shape keeps the default config.
//
// 1. Narrow single-row contractions (`match_single_row_vecmat`). The compiler
//    default gives a rank-1 vecmat `thread = [8, 0]`, `workgroup = [256, 0]`
//    for every width at or above 2048, so the launch has N/256 workgroups of
//    one warp each: 8 for the 2048-wide q and o and 32 for gate/up on
//    llama-3.2-1b, 14 for the 3584-wide q, o and down projections of a model
//    with hidden 3584. On a 20-WGP device that is less than one wave, and the
//    serial K reduction cannot hide memory latency, so llama's q/o run at
//    about 42 GB/s where the 128256-wide LM head reaches 228 GB/s. One output
//    per thread raises the launch to N/32 warps and, with
//    `reduction = [0, 32]`, measures llama's q/o at about 219 GB/s and its
//    K = 8192 down projection at about 230 (issue #2873). The tile does not
//    depend on the reduction length: every output still walks K serially and
//    only the output count decides the occupancy, so the match is any K with
//    128 <= N <= 8192, a rank-1 by rank-2 contraction with f16 operands and an
//    f32 accumulator and no batch dims. The N bound keeps the LM head (128256
//    wide on llama) out, where the activation re-read would grow with N, and
//    the N >= 128 bound keeps very narrow projections on the compiler default.
//
// 2. Wide long-K single-row contractions (`match_bigk_vecmat_wide`). The
//    previous long-K config for N >= 8193 and K >= 4096, where the default
//    thread tile is left alone because its activation re-read would grow with
//    N. This is the LM head of a checkpoint whose hidden size is at least 4096.
//
// 3. The grouped-query decode score contraction (`match_gqa_score`). It is
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
//
// 4. The grouped-query decode context contraction (`match_gqa_context`). It is
//    [batch = n_kv, M = group, K = keys] by [batch, K, N = head_dim] f16 into
//    an f32 result, so M is 4, the reduction is the key axis (the context
//    capacity) and N is 64 on llama-3.2-1b. It has the same batched layout as
//    the score contraction with the operands' roles swapped, and does more
//    multiply-accumulates, yet the compiler default leaves it at about 6.4 us
//    per dispatch at capacity 256, four times the tuned score dispatch. The
//    default walks one workgroup per output element (grid 256 by 8, 256
//    threads, no reuse across the 4 query rows). The config below tiles all
//    four query rows and the whole 64-wide output per workgroup, steps the key
//    reduction in 64-element chunks and computes two output columns per
//    thread; the dispatch drops to about 1.8 us at capacity 256, 2.8 us at 512
//    and 5.9 us at 1024 (issue #2956). The match is as narrow as the score one
//    for the same reasons: batch = n_kv = 8, M = 4 and N = 64 exactly, key axis
//    at least 128. The batch predicate keeps the ragged and uniform-B graphs
//    (collapsed batch = bsz * n_kv) out, and prefill keeps out through its M
//    axis, which carries the prompt length. Other head widths, group sizes or
//    key counts keep the compiler default until they are measured.
module attributes {iree_codegen.tuning_spec_with_default_entrypoint, transform.with_named_sequence} {
  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}, %config: !transform.any_param {transform.readonly}) {
    transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_single_row_vecmat(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %batch_dims, %m_dims, %n_dims, %k_dims = transform.iree.match.contraction %arg0,
      lhs_type = f16, rhs_type = f16, output_type = f32,
      indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>]
      : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch_dims, [] : !transform.param<i64>
    %rhs = transform.get_operand %arg0[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %rhs[1], umin = 128, umax = 8192 : !transform.any_value
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{reduction = [0, 32], thread = [1, 0], workgroup = [32, 0]}>, translation_info = <pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [32, 1, 1] subgroup_size = 32>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_bigk_vecmat_wide(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %batch_dims, %m_dims, %n_dims, %k_dims = transform.iree.match.contraction %arg0,
      lhs_type = f16, rhs_type = f16, output_type = f32,
      indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>]
      : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch_dims, [] : !transform.param<i64>
    %rhs = transform.get_operand %arg0[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %rhs[0], umin = 4096, none : !transform.any_value
    transform.iree.match.dim_bounds %rhs[1], umin = 8193, none : !transform.any_value
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

  transform.named_sequence @match_gqa_context(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %batch_dims, %m_dims, %n_dims, %k_dims = transform.iree.match.contraction %arg0,
      lhs_type = f16, rhs_type = f16, output_type = f32,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
      : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch_dims, [8] : !transform.param<i64>
    transform.iree.match.dims_equal %m_dims, [4] : !transform.param<i64>
    transform.iree.match.dims_equal %n_dims, [64] : !transform.param<i64>
    %rhs = transform.get_operand %arg0[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %rhs[1], umin = 128, none : !transform.any_value
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{reduction = [0, 0, 0, 64], thread = [1, 1, 2, 0], workgroup = [1, 4, 64, 1]}>, translation_info = <pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [32, 4, 1] subgroup_size = 32>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed}) -> !transform.any_op
      attributes {iree_codegen.tuning_spec_entrypoint} {
    %updated = transform.foreach_match in %arg0
      @match_single_row_vecmat -> @apply_op_config,
      @match_bigk_vecmat_wide -> @apply_op_config,
      @match_gqa_score -> @apply_op_config,
      @match_gqa_context -> @apply_op_config
      : (!transform.any_op) -> !transform.any_op
    transform.yield %updated : !transform.any_op
  }
}
