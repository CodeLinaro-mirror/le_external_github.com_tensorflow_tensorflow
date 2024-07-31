

/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/slice.h"

#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {
namespace {

class LegalizeSliceOp : public OpConversionPattern<mhlo::SliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::SliceOp slice_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto begin = rewriter.create<arith::ConstantOp>(slice_op.getLoc(),
                                                    slice_op.getStartIndices());
    auto end = rewriter.create<arith::ConstantOp>(slice_op.getLoc(),
                                                  slice_op.getLimitIndices());
    auto strides = rewriter.create<arith::ConstantOp>(slice_op.getLoc(),
                                                      slice_op.getStrides());
    auto zero = IntegerAttr::get(rewriter.getI32Type(), 0);
    auto no_offset = BoolAttr::get(rewriter.getContext(), false);

    auto begin_cast_type = RankedTensorType::get(
        llvm::cast<ShapedType>(begin.getType()).getShape(),
        rewriter.getI32Type());
    auto begin_cast =
        rewriter.create<TFL::CastOp>(slice_op.getLoc(), begin_cast_type, begin);

    auto end_cast_type =
        RankedTensorType::get(llvm::cast<ShapedType>(end.getType()).getShape(),
                              rewriter.getI32Type());
    auto end_cast =
        rewriter.create<TFL::CastOp>(slice_op.getLoc(), end_cast_type, end);

    auto strides_cast_type = RankedTensorType::get(
        llvm::cast<ShapedType>(strides.getType()).getShape(),
        rewriter.getI32Type());
    auto strides_cast = rewriter.create<TFL::CastOp>(
        slice_op.getLoc(), strides_cast_type, strides);

    auto strided_slice_op = rewriter.create<TFL::StridedSliceOp>(
        slice_op.getLoc(), slice_op.getType(), slice_op.getOperand(),
        begin_cast, end_cast, strides_cast, zero, zero, zero, zero, zero,
        no_offset);
    strided_slice_op->dump();

    rewriter.replaceOp(slice_op, strided_slice_op);
    return success();
  }
};

}  // namespace

void PopulateSlicePatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                           ConversionTarget& target) {
  patterns.add<LegalizeSliceOp>(ctx);
  target.addIllegalOp<mhlo::SliceOp>();
}

}  // namespace mlir::odml
