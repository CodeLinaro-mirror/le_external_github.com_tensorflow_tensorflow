// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-flatten-tensors | FileCheck %s

func.func @tensor_extract(
    %arg0: tensor<2x3xf32, dense<[0, 1]> : tensor<2xi64>>,
    %arg1: index, %arg2: index) -> f32 {
  %v = tensor.extract %arg0[%arg1, %arg2]
      : tensor<2x3xf32, dense<[0, 1]> : tensor<2xi64>>
  func.return %v : f32
}
// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d1 * 2 + d0)>

// CHECK-LABEL: func.func @tensor_extract(
// CHECK-SAME:      %[[SRC:.*]]: tensor<6xf32>,
// CHECK-SAME:      %[[I:.*]]: index, %[[J:.*]]: index) -> f32 {
// CHECK:        %[[INDEX:.*]] = xla_gpu.apply_indexing #[[$MAP]](%[[I]]
// CHECK-SAME:     in [0, 2), %[[J]] in [0, 3))
// CHECK:        tensor.extract %[[SRC]][%[[INDEX]]] : tensor<6xf32>

// -----

func.func @tensor_insert(
    %arg0: tensor<10x24xcomplex<f32>>) -> tensor<10x24xcomplex<f32>> {
  %c1 = arith.constant 1 : index
  %real = arith.constant 3.0 : f32
  %imag = arith.constant 2.0 : f32
  %complex = complex.create %real, %imag : complex<f32>
  %out = tensor.insert %complex into %arg0[%c1, %c1] : tensor<10x24xcomplex<f32>>
  func.return %out : tensor<10x24xcomplex<f32>>
}
// CHECK-LABEL: func.func @tensor_insert(
// CHECK-SAME:      %[[TENSOR:.*]]: tensor<240xcomplex<f32>>) -> tensor<240xcomplex<f32>> {
// CHECK:         %[[INDEX:.*]] = arith.constant 25
// CHECK:         %[[COMPLEX:.*]] = complex.create
// CHECK:         tensor.insert %[[COMPLEX]] into %[[TENSOR]][%[[INDEX]]]
// CHECK-SAME:      : tensor<240xcomplex<f32>>

// -----

func.func @atomic_rmw(%in: tensor<2x4xf32>, %i: index, %j: index)
    -> (tensor<2x4xf32>) {
  %ret = xla_gpu.atomic_rmw %in[%i, %j] : tensor<2x4xf32> {
    ^bb0(%current : f32):
      %c42 = arith.constant 1.0 : f32
      %add = arith.minimumf %current, %c42 : f32
      xla_gpu.yield %add : f32
  }
  return %ret : tensor<2x4xf32>
}
// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>

// CHECK-LABEL: func.func @atomic_rmw(
// CHECK-SAME:      %[[TENSOR:.*]]: tensor<8xf32>, %[[I:.*]]: index,
// CHECK-SAME:      %[[J:.*]]: index) -> tensor<8xf32> {
// CHECK:         %[[INDEX:.*]] = xla_gpu.apply_indexing #[[$MAP]]
// CHECK-SAME:      (%[[I]] in [0, 2), %[[J]] in [0, 4))
// CHECK:         xla_gpu.atomic_rmw %[[TENSOR]][%[[INDEX]]] : tensor<8xf32>
