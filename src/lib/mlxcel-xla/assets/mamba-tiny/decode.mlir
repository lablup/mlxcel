module @decode_step {
  util.global private mutable @ssm_h : tensor<1x16x4xf32>
  util.global private mutable @ssm_conv : tensor<1x3x16xf32>
  func.func public @main(%arg0: tensor<12x8xf32> loc("params['embed']"), %arg1: tensor<8xf32> loc("params['final_norm']"), %arg2: tensor<12x8xf32> loc("params['lm_head']"), %arg3: tensor<8xf32> loc("params['layers'][0]['in_ln']"), %arg4: tensor<32x8xf32> loc("params['layers'][0]['ssm.in_proj']"), %arg5: tensor<32xf32> loc("params['layers'][0]['ssm.in_proj_b']"), %arg6: tensor<16x1x4xf32> loc("params['layers'][0]['ssm.conv_w']"), %arg7: tensor<16xf32> loc("params['layers'][0]['ssm.conv_b']"), %arg8: tensor<10x16xf32> loc("params['layers'][0]['ssm.x_proj']"), %arg9: tensor<16x2xf32> loc("params['layers'][0]['ssm.dt_proj']"), %arg10: tensor<16xf32> loc("params['layers'][0]['ssm.dt_proj_b']"), %arg11: tensor<16x4xf32> loc("params['layers'][0]['ssm.a_log']"), %arg12: tensor<16xf32> loc("params['layers'][0]['ssm.d']"), %arg13: tensor<8x16xf32> loc("params['layers'][0]['ssm.out_proj']"), %arg14: tensor<8xf32> loc("params['layers'][0]['ssm.out_proj_b']"), %arg15: i32 loc("token"), %arg16: i32 loc("pos"), %arg17: i32 loc("cache_len")) -> tensor<12xf32> {
    %0 = tensor.from_elements %arg15 : tensor<i32>
    %1 = tensor.from_elements %arg16 : tensor<i32>
    %2 = tensor.from_elements %arg17 : tensor<i32>
    %3 = util.global.load @ssm_h : tensor<1x16x4xf32>
    %4 = util.global.load @ssm_conv : tensor<1x3x16xf32>
    %5 = stablehlo.slice %3 [0:1, 0:16, 0:4] : (tensor<1x16x4xf32>) -> tensor<1x16x4xf32>
    %6 = stablehlo.slice %4 [0:1, 0:3, 0:16] : (tensor<1x3x16xf32>) -> tensor<1x3x16xf32>
    %7 = stablehlo.constant dense<0x00000000> : tensor<f32>
    %8 = stablehlo.constant dense<0x3F800000> : tensor<f32>
    %9 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %10 = stablehlo.constant dense<0xF149F2CA> : tensor<f32>
    %11 = stablehlo.constant dense<0x3727C5AC> : tensor<f32>
    %12 = stablehlo.constant dense<0x41000000> : tensor<f32>
    %13 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %14 = stablehlo.constant dense<0> : tensor<i32>
    %15 = stablehlo.constant dense<0> : tensor<i32>
    %16 = stablehlo.dynamic_slice %arg0, %0, %14, sizes = [1, 8] : (tensor<12x8xf32>, tensor<i32>, tensor<i32>) -> tensor<1x8xf32>
    %17 = stablehlo.reshape %16 : (tensor<1x8xf32>) -> tensor<8xf32>
    %18 = stablehlo.multiply %17, %17 : tensor<8xf32>
    %19 = stablehlo.reduce(%18 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<f32>
    %20 = stablehlo.divide %19, %12 : tensor<f32>
    %21 = stablehlo.add %20, %11 : tensor<f32>
    %22 = stablehlo.rsqrt %21 : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %24 = stablehlo.multiply %17, %23 : tensor<8xf32>
    %25 = stablehlo.multiply %24, %arg3 : tensor<8xf32>
    %26 = stablehlo.reshape %25 : (tensor<8xf32>) -> tensor<1x8xf32>
    %27 = stablehlo.dot_general %26, %arg4, contracting_dims = [1] x [1] : (tensor<1x8xf32>, tensor<32x8xf32>) -> tensor<1x32xf32>
    %28 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<32xf32>) -> tensor<1x32xf32>
    %29 = stablehlo.add %27, %28 : tensor<1x32xf32>
    %30 = stablehlo.slice %29 [0:1, 0:16] : (tensor<1x32xf32>) -> tensor<1x16xf32>
    %31 = stablehlo.slice %29 [0:1, 16:32] : (tensor<1x32xf32>) -> tensor<1x16xf32>
    %32 = stablehlo.reshape %30 : (tensor<1x16xf32>) -> tensor<1x1x16xf32>
    %33 = stablehlo.concatenate %6, %32, dim = 1 : (tensor<1x3x16xf32>, tensor<1x1x16xf32>) -> tensor<1x4x16xf32>
    %34 = stablehlo.reshape %arg6 : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %35 = stablehlo.transpose %34, dims = [1, 0] : (tensor<16x4xf32>) -> tensor<4x16xf32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [1, 2] : (tensor<4x16xf32>) -> tensor<1x4x16xf32>
    %37 = stablehlo.multiply %33, %36 : tensor<1x4x16xf32>
    %38 = stablehlo.constant dense<0x00000000> : tensor<f32>
    %39 = stablehlo.reduce(%37 init: %38) applies stablehlo.add across dimensions = [1] : (tensor<1x4x16xf32>, tensor<f32>) -> tensor<1x16xf32>
    %40 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %41 = stablehlo.add %39, %40 : tensor<1x16xf32>
    %42 = stablehlo.slice %33 [0:1, 1:4, 0:16] : (tensor<1x4x16xf32>) -> tensor<1x3x16xf32>
    %43 = stablehlo.negate %41 : tensor<1x16xf32>
    %44 = stablehlo.exponential %43 : tensor<1x16xf32>
    %45 = stablehlo.constant dense<0x3F800000> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %45, dims = [] : (tensor<f32>) -> tensor<1x16xf32>
    %47 = stablehlo.add %46, %44 : tensor<1x16xf32>
    %48 = stablehlo.divide %46, %47 : tensor<1x16xf32>
    %49 = stablehlo.multiply %41, %48 : tensor<1x16xf32>
    %50 = stablehlo.dot_general %49, %arg8, contracting_dims = [1] x [1] : (tensor<1x16xf32>, tensor<10x16xf32>) -> tensor<1x10xf32>
    %51 = stablehlo.slice %50 [0:1, 0:2] : (tensor<1x10xf32>) -> tensor<1x2xf32>
    %52 = stablehlo.slice %50 [0:1, 2:6] : (tensor<1x10xf32>) -> tensor<1x4xf32>
    %53 = stablehlo.slice %50 [0:1, 6:10] : (tensor<1x10xf32>) -> tensor<1x4xf32>
    %54 = stablehlo.dot_general %51, %arg9, contracting_dims = [1] x [1] : (tensor<1x2xf32>, tensor<16x2xf32>) -> tensor<1x16xf32>
    %55 = stablehlo.broadcast_in_dim %arg10, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %56 = stablehlo.add %54, %55 : tensor<1x16xf32>
    %57 = stablehlo.constant dense<0x00000000> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %57, dims = [] : (tensor<f32>) -> tensor<1x16xf32>
    %59 = stablehlo.maximum %56, %58 : tensor<1x16xf32>
    %60 = stablehlo.abs %56 : tensor<1x16xf32>
    %61 = stablehlo.negate %60 : tensor<1x16xf32>
    %62 = stablehlo.exponential %61 : tensor<1x16xf32>
    %63 = stablehlo.log_plus_one %62 : tensor<1x16xf32>
    %64 = stablehlo.add %59, %63 : tensor<1x16xf32>
    %65 = stablehlo.exponential %arg11 : tensor<16x4xf32>
    %66 = stablehlo.negate %65 : tensor<16x4xf32>
    %67 = stablehlo.reshape %64 : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %68 = stablehlo.broadcast_in_dim %67, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x4xf32>
    %69 = stablehlo.broadcast_in_dim %66, dims = [1, 2] : (tensor<16x4xf32>) -> tensor<1x16x4xf32>
    %70 = stablehlo.multiply %68, %69 : tensor<1x16x4xf32>
    %71 = stablehlo.multiply %64, %49 : tensor<1x16xf32>
    %72 = stablehlo.reshape %71 : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x4xf32>
    %74 = stablehlo.reshape %52 : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
    %75 = stablehlo.broadcast_in_dim %74, dims = [0, 1, 2] : (tensor<1x1x4xf32>) -> tensor<1x16x4xf32>
    %76 = stablehlo.multiply %73, %75 : tensor<1x16x4xf32>
    %77 = stablehlo.exponential %70 : tensor<1x16x4xf32>
    %78 = stablehlo.multiply %77, %5 : tensor<1x16x4xf32>
    %79 = stablehlo.add %78, %76 : tensor<1x16x4xf32>
    %80 = stablehlo.reshape %53 : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2] : (tensor<1x1x4xf32>) -> tensor<1x16x4xf32>
    %82 = stablehlo.multiply %79, %81 : tensor<1x16x4xf32>
    %83 = stablehlo.constant dense<0x00000000> : tensor<f32>
    %84 = stablehlo.reduce(%82 init: %83) applies stablehlo.add across dimensions = [2] : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<1x16xf32>
    %85 = stablehlo.broadcast_in_dim %arg12, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %86 = stablehlo.multiply %49, %85 : tensor<1x16xf32>
    %87 = stablehlo.add %84, %86 : tensor<1x16xf32>
    %88 = stablehlo.negate %31 : tensor<1x16xf32>
    %89 = stablehlo.exponential %88 : tensor<1x16xf32>
    %90 = stablehlo.constant dense<0x3F800000> : tensor<f32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<f32>) -> tensor<1x16xf32>
    %92 = stablehlo.add %91, %89 : tensor<1x16xf32>
    %93 = stablehlo.divide %91, %92 : tensor<1x16xf32>
    %94 = stablehlo.multiply %31, %93 : tensor<1x16xf32>
    %95 = stablehlo.multiply %87, %94 : tensor<1x16xf32>
    %96 = stablehlo.dot_general %95, %arg13, contracting_dims = [1] x [1] : (tensor<1x16xf32>, tensor<8x16xf32>) -> tensor<1x8xf32>
    %97 = stablehlo.broadcast_in_dim %arg14, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %98 = stablehlo.add %96, %97 : tensor<1x8xf32>
    %99 = stablehlo.reshape %98 : (tensor<1x8xf32>) -> tensor<8xf32>
    %100 = stablehlo.add %17, %99 : tensor<8xf32>
    %101 = stablehlo.multiply %100, %100 : tensor<8xf32>
    %102 = stablehlo.reduce(%101 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<f32>
    %103 = stablehlo.divide %102, %12 : tensor<f32>
    %104 = stablehlo.add %103, %11 : tensor<f32>
    %105 = stablehlo.rsqrt %104 : tensor<f32>
    %106 = stablehlo.broadcast_in_dim %105, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %107 = stablehlo.multiply %100, %106 : tensor<8xf32>
    %108 = stablehlo.multiply %107, %arg1 : tensor<8xf32>
    %109 = stablehlo.dot_general %108, %arg2, contracting_dims = [0] x [1] : (tensor<8xf32>, tensor<12x8xf32>) -> tensor<12xf32>
    %110 = stablehlo.constant dense<0> : tensor<i32>
    %111 = stablehlo.constant dense<0> : tensor<i32>
    %112 = stablehlo.reshape %79 : (tensor<1x16x4xf32>) -> tensor<1x16x4xf32>
    %113 = stablehlo.reshape %42 : (tensor<1x3x16xf32>) -> tensor<1x3x16xf32>
    %114 = stablehlo.dynamic_update_slice %3, %112, %111, %110, %110 : (tensor<1x16x4xf32>, tensor<1x16x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x16x4xf32>
    %115 = stablehlo.dynamic_update_slice %4, %113, %111, %110, %110 : (tensor<1x3x16xf32>, tensor<1x3x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x3x16xf32>
    util.global.store %114, @ssm_h : tensor<1x16x4xf32>
    util.global.store %115, @ssm_conv : tensor<1x3x16xf32>
    return %109 : tensor<12xf32>
  }
  func.func public @state_import(%s0: tensor<1x16x4xf32>, %s1: tensor<1x3x16xf32>) {
    util.global.store %s0, @ssm_h : tensor<1x16x4xf32>
    util.global.store %s1, @ssm_conv : tensor<1x3x16xf32>
    return
  }
  func.func public @scalar_inputs() {
    return
  }
}
