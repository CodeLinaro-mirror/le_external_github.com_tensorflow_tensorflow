// RUN: mlir-hlo-opt --stablehlo-legalize-to-hlo=partial-conversion=true %s | FileCheck %s --check-prefix CHECK-STABLEHLO-OP-UNCHANGED
// RUN: hlo-translate -mlir-to-hlo %s | FileCheck %s

// Validate stablehlo -> hlo direct conversion path works fine.
// Do not extend it for all stablehlo ops.
func.func @main(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK-STABLEHLO-OP-UNCHANGED: stablehlo.constant
  // CHECK: %[[ARG0:.*]] = s32[] parameter(0)
  // CHECK: %[[C:.*]] = s32[] constant(2)
  %c = stablehlo.constant dense<2> : tensor<i32>
  // CHECK-STABLEHLO-OP-UNCHANGED: stablehlo.add
  // CHECK: %[[ADD:.*]] = s32[] add(%[[ARG0]], %[[C]])
  %0 = stablehlo.add %arg0, %c : tensor<i32>
  // CHECK: s32[] multiply(%[[ADD]], %[[C]])
  %1 = mhlo.multiply %c, %0 : tensor<i32>
  return %1 : tensor<i32>
}
