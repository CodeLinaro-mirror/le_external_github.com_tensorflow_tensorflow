// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-lower-tensors="target_type=cpu" \
// RUN: | FileCheck %s

func.func @load_non_gep_from_args(%arg0: !llvm.ptr) -> !llvm.ptr {
  %0 = llvm.getelementptr inbounds %arg0[1]
    : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
  %1 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
  %2 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
  func.return %2 : !llvm.ptr
}

// CHECK-LABEL: @load_non_gep_from_args
// CHECK-NEXT:    %0 = llvm.getelementptr inbounds %arg0[1]
// CHECK-NEXT:    %1 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %2 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    return %2 : !llvm.ptr

// -----

func.func @transfer_read_alignment(%arg0: tensor<8xi64> {llvm.align = 32 : index}) -> vector<8xi64> {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %0 = vector.transfer_read %arg0[%c0], %c0_i64 {in_bounds = [true]} : tensor<8xi64>, vector<8xi64>
  return %0 : vector<8xi64>
}
// CHECK-LABEL: @transfer_read_alignment(
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr
// CHECK:           %[[LOADED:.*]] = llvm.load %[[ARG0]] {alignment = 32 : i64} : !llvm.ptr
// CHECK:           return %[[LOADED]] : vector<8xi64>

// -----

func.func @transfer_read_alignment_non_zero_index(%arg0: tensor<16xi64> {llvm.align = 32 : index}) -> vector<8xi64> {
  %c8 = arith.constant 8 : index
  %c0_i64 = arith.constant 0 : i64
  %0 = vector.transfer_read %arg0[%c8], %c0_i64 {in_bounds = [true]} : tensor<16xi64>, vector<8xi64>
  return %0 : vector<8xi64>
}
// CHECK-LABEL: @transfer_read_alignment_non_zero_index(
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr
// CHECK:           %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][8]
// CHECK-NEXT:      llvm.load %[[PTR]] : !llvm.ptr -> vector<8xi64>
