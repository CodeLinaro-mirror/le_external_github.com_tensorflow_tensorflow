// RUN: hlo-translate -mlir-to-hlo %s | FileCheck %s

// Tests for all stablehlo ops to validate stablehlo -> hlo conversion. 

func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[ARG0:.*]] = f32[4] parameter(0)
  // CHECK: %[[ARG1:.*]] = f32[4] parameter(1)
  // CHECK: ROOT %add.3 = f32[4] add(%[[ARG0]], %[[ARG1]])
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>  func.return %0 : tensor<4xf32>
}
