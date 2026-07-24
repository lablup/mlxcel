module @aux_smoke {
  func.func public @main(
      %weight: tensor<2xf32>,
      %floats: tensor<2xf32>,
      %integers: tensor<2xi32>,
      %mask: tensor<2xi1>
  ) -> (tensor<2xf32>, tensor<2xi32>, tensor<2xi1>) {
    %sum = stablehlo.add %weight, %floats : tensor<2xf32>
    return %sum, %integers, %mask : tensor<2xf32>, tensor<2xi32>, tensor<2xi1>
  }
}
