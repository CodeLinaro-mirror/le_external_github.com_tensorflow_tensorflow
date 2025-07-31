// RUN: hlo-translate -mlir-to-hlo -split-input-file %s | FileCheck %s

// CHECK-LABEL: ENTRY %main.{{.*}} (Arg_0.1: f32[4], Arg_1.2: f32[4]) -> f32[]
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<f32> {
  // CHECK: %Arg_0.1 = f32[4] parameter(0)
  // CHECK: %Arg_1.2 = f32[4] parameter(1)
  // CHECK: %add.3 = f32[4] add(%Arg_0.1, %Arg_1.2)
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
  // CHECK: ROOT %dot.4 = f32[] dot(%add.3, %Arg_1.2), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  %1 = stablehlo.dot %0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----
// MHLO to HLO

// CHECK-LABEL: ENTRY %main.{{.*}} (Arg_0.1: f32[4], Arg_1.2: f32[4]) -> f32[4]
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %Arg_0.1 = f32[4] parameter(0)
  // CHECK-NEXT: %Arg_1.2 = f32[4] parameter(1)
  // CHECK-NEXT: %add.3 = f32[4] add(%Arg_0.1, %Arg_1.2)
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: ROOT %add.4 = f32[4] add(%add.3, %Arg_1.2)
  %1 = "mhlo.add"(%0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// -----
// CHECK: HloModule
// CHECK: ENTRY
// CHECK-NOT: manual
func.func @main() -> tensor<2048x128xf32> {
  // CHECK-NEXT: %after-all.1 = token[] after-all(), sharding={manual}
  // CHECK-NEXT: %infeed.2 = ((f32[2048,128]), token[]) infeed(%after-all.1), sharding={{[{][{]manual}, {manual[}][}]}}
  // CHECK-NEXT: %get-tuple-element.3 = (f32[2048,128]) get-tuple-element(%infeed.2), index=0, sharding={{[{][{]manual[}][}]}}
  // CHECK-NEXT: ROOT %get-tuple-element.4 = f32[2048,128] get-tuple-element(%get-tuple-element.3), index=0, sharding={manual}
  // CHECK-NEXT: %get-tuple-element.5 = token[] get-tuple-element(%infeed.2), index=1, sharding={manual}
  %56 = mhlo.create_token {mhlo.sharding = "{manual}", xla_shape = "token[]"} : !mhlo.token
  %57:2 = "mhlo.infeed"(%56) <{infeed_config = "", layout = [[1, 0]]}> {mhlo.sharding = "{{manual}, {manual}}"} : (!mhlo.token) -> (tensor<2048x128xf32>, !mhlo.token)
  return %57#0 : tensor<2048x128xf32>
}

// -----
// CHECK: HloModule
// CHECK: ENTRY
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> () {
  // CHECK: %tuple.6 = (f32[4], f32[4]) tuple(%add.3, %add.4), sharding={{[{][{]manual}, {manual[}][}]}}
  // CHECK-NEXT: %after-all.5 = token[] after-all(), sharding={manual}
  // CHECK-NEXT:  %outfeed.7 = token[] outfeed(%tuple.6, %after-all.5), outfeed_shape=(f32[4]{0}, f32[4]{0}), sharding={manual}
  // CHECK-NEXT: ROOT %tuple.8 = () tuple()
  %0 = "mhlo.add"(%arg0, %arg1) {mhlo.sharding = "{manual}"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "mhlo.add"(%arg1, %arg0) {mhlo.sharding = "{manual}"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %56 = mhlo.create_token {mhlo.sharding = "{manual}", xla_shape = "token[]"} : !mhlo.token
  %57 = "mhlo.outfeed"(%0, %1, %56) <{infeed_config = "", layout = [[1, 0]]}> {mhlo.sharding = "{manual}"} : (tensor<4xf32>, tensor<4xf32>, !mhlo.token) -> !mhlo.token
  return
}

// -----
// CHECK: HloModule
// CHECK: ENTRY
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> () {
  // CHECK: %tuple.6 = (f32[4], f32[4]) tuple(%add.3, %add.4), sharding={{[{][{]manual}, {replicated[}][}]}}
  // CHECK-NEXT: %after-all.5 = token[] after-all(), sharding={manual}
  // CHECK-NEXT:  %outfeed.7 = token[] outfeed(%tuple.6, %after-all.5), outfeed_shape=(f32[4]{0}, f32[4]{0}), sharding={manual}
  // CHECK-NEXT: ROOT %tuple.8 = () tuple()
  %0 = "mhlo.add"(%arg0, %arg1) {mhlo.sharding = "{manual}"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "mhlo.add"(%arg1, %arg0) {} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %56 = mhlo.create_token {mhlo.sharding = "{manual}", xla_shape = "token[]"} : !mhlo.token
  %57 = "mhlo.outfeed"(%0, %1, %56) <{infeed_config = "", layout = [[1, 0]]}> {mhlo.sharding = "{manual}"} : (tensor<4xf32>, tensor<4xf32>, !mhlo.token) -> !mhlo.token
  return
}
