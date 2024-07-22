/* Copyright 2024 The OpenXLA Authors.

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
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {
namespace {

#define GEN_PASS_DEF_FLATTENTENSORSPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::SmallVector;
using mlir::Type;
using mlir::TypedValue;
using mlir::TypeRange;
using mlir::UnrealizedConversionCastOp;
using mlir::Value;
using mlir::ValueRange;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
using mlir::tensor::ExtractOp;
using mlir::tensor::InsertOp;

RankedTensorType GetFlattenedType(RankedTensorType tensor_type) {
  return RankedTensorType::get({tensor_type.getNumElements()},
                               tensor_type.getElementType());
}

bool HasOnlyFlatTensorsOrScalars(TypeRange types) {
  return llvm::all_of(types, [](Type ty) {
    auto tensor_type = mlir::dyn_cast<RankedTensorType>(ty);
    if (!tensor_type) return true;
    return tensor_type.getRank() < 2;
  });
}

struct RewriteFunctionSignatures : OpRewritePattern<FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter& rewriter) const override {
    auto input_types = op.getFunctionType().getInputs();
    auto result_types = op.getFunctionType().getResults();
    if (HasOnlyFlatTensorsOrScalars(input_types) &&
        HasOnlyFlatTensorsOrScalars(result_types)) {
      return rewriter.notifyMatchFailure(op, "nothing to flatten");
    }

    bool some_tensor_results = llvm::any_of(result_types, [](Type ty) {
      return mlir::isa<mlir::RankedTensorType>(ty);
    });
    bool some_scalar_results = llvm::all_of(result_types, [](Type ty) {
      return !mlir::isa<mlir::RankedTensorType>(ty);
    });
    if (some_tensor_results && some_scalar_results) {
      op->emitOpError("function has a mix of tensor and non-tensor results");
      return mlir::failure();
    }
    auto loc = op.getLoc();

    mlir::Block* entry_block = &op.getBody().front();
    SmallVector<Type> new_result_types;
    SmallVector<Value> new_results;
    if (some_tensor_results) {
      // If some results are tensors, we need to flatten them.
      auto terminator = entry_block->getTerminator();
      rewriter.setInsertionPoint(terminator);

      for (Value result : terminator->getOperands()) {
        auto tensor_type = mlir::cast<RankedTensorType>(result.getType());
        auto new_result_type = GetFlattenedType(tensor_type);
        new_result_types.push_back(new_result_type);

        Value result_1d = rewriter
                              .create<UnrealizedConversionCastOp>(
                                  loc, new_result_type, result)
                              .getResult(0);
        new_results.push_back(result_1d);
      }
      rewriter.replaceOpWithNewOp<ReturnOp>(terminator, new_results);
    } else {
      // If all results are scalars, we don't need to do anything.
      new_result_types.append(result_types.begin(), result_types.end());
    }

    // Cast all function arguments to the original type.
    SmallVector<Type> new_operand_types(input_types);
    rewriter.setInsertionPointToStart(entry_block);
    for (auto&& [index, operand_type] : llvm::enumerate(new_operand_types)) {
      if (auto tensor_type = mlir::dyn_cast<RankedTensorType>(operand_type)) {
        mlir::BlockArgument func_argument = op.getArgument(index);
        auto cast_to_orig_type = rewriter.create<UnrealizedConversionCastOp>(
            loc, operand_type, func_argument);
        func_argument.replaceAllUsesExcept(cast_to_orig_type.getResult(0),
                                           cast_to_orig_type);
        operand_type = GetFlattenedType(tensor_type);
      }
    }
    // Replace the function arguments with the new types.
    for (auto [arg, arg_type] :
         llvm::zip(entry_block->getArguments(), new_operand_types)) {
      arg.setType(arg_type);
    }
    // Update function signature.
    op.setType(rewriter.getFunctionType(new_operand_types, new_result_types));
    return mlir::success();
  }
};

// Returns the lineariazed index, if the rank is greater than 1. Otherwise,
// returns nullptr.
Value LinearizeIndex(TypedValue<mlir::RankedTensorType> tensor,
                     ValueRange indices, PatternRewriter& rewriter) {
  if (tensor.getType().getRank() < 2) {
    return nullptr;
  }
  auto byte_shape = ShapeUtil::MakeShape(U8, tensor.getType().getShape());
  if (auto encoding = tensor.getType().getEncoding()) {
    *byte_shape.mutable_layout() = LayoutUtil::MakeLayout(llvm::to_vector(
        mlir::cast<mlir::DenseElementsAttr>(encoding).getValues<int64_t>()));
  }
  auto linear_shape =
      ShapeUtil::MakeShape(U8, {ShapeUtil::ElementsIn(byte_shape)});
  auto linearized_map =
      GetBitcastMap(byte_shape, linear_shape, tensor.getContext());
  mlir::SmallVector<Value> result;
  rewriter.createOrFold<ApplyIndexingOp>(result, tensor.getLoc(), indices,
                                         ValueRange{}, linearized_map);
  return result.front();
}

struct RewriteTensorExtract : OpRewritePattern<ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor = op.getTensor();
    auto tensor_type = tensor.getType();
    auto linear_index = LinearizeIndex(tensor, op.getIndices(), rewriter);
    if (linear_index == nullptr) {
      return rewriter.notifyMatchFailure(op, "the tensor is already flat");
    }
    auto tensor_1D = rewriter
                         .create<UnrealizedConversionCastOp>(
                             op.getLoc(), GetFlattenedType(tensor_type), tensor)
                         .getResult(0);
    rewriter.replaceOpWithNewOp<ExtractOp>(op, tensor_1D, linear_index);
    return mlir::success();
  }
};

struct RewriteTensorInsert : OpRewritePattern<InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor = op.getDest();
    auto tensor_type = tensor.getType();
    auto linear_index = LinearizeIndex(tensor, op.getIndices(), rewriter);
    if (linear_index == nullptr) {
      return rewriter.notifyMatchFailure(op, "the tensor is already flat");
    }
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto tensor_1D = b.create<UnrealizedConversionCastOp>(
                          GetFlattenedType(tensor_type), tensor)
                         .getResult(0);
    auto new_insert =
        b.create<InsertOp>(op.getScalar(), tensor_1D, linear_index);
    auto cast_to_orig_type = b.create<UnrealizedConversionCastOp>(
        tensor_type, new_insert.getResult());
    rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
    return mlir::success();
  }
};

struct RewriteAtomicRMW : OpRewritePattern<AtomicRMWOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AtomicRMWOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor = op.getInput();
    auto tensor_type = tensor.getType();
    auto linear_index = LinearizeIndex(tensor, op.getIndices(), rewriter);
    if (linear_index == nullptr) {
      return rewriter.notifyMatchFailure(op, "the tensor is already flat");
    }
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto tensor_1D = b.create<UnrealizedConversionCastOp>(
                          GetFlattenedType(tensor_type), tensor)
                         .getResult(0);
    auto new_atomic_rmw = b.create<AtomicRMWOp>(tensor_1D, linear_index);
    rewriter.inlineRegionBefore(op.getRegion(),
                                &new_atomic_rmw.getRegion().front());
    auto cast_to_orig_type = b.create<UnrealizedConversionCastOp>(
        tensor_type, new_atomic_rmw.getResult());
    rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
    return mlir::success();
  }
};

class FlattenTensorsPass
    : public impl::FlattenTensorsPassBase<FlattenTensorsPass> {
 public:
  void runOnOperation() override {
    MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    // clang-format off
    patterns.add<
        RewriteAtomicRMW,
        RewriteFunctionSignatures,
        RewriteTensorExtract,
        RewriteTensorInsert
    >(mlir_context);
    // clang-format on
    ApplyIndexingOp::getCanonicalizationPatterns(patterns, mlir_context);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateFlattenTensorsPass() {
  return std::make_unique<FlattenTensorsPass>();
}

}  // namespace gpu
}  // namespace xla
