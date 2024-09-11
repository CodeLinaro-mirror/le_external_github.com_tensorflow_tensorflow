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

#include "xla/service/spmd/shardy/sdy_round_trip/shard_map_export.h"

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::NamedAttribute;
using ::mlir::StringRef;
using ::mlir::func::FuncOp;

namespace stablehlo = ::mlir::stablehlo;
namespace sdy = ::mlir::sdy;

class SdyRoundTripShardMapExportPass
    : public mlir::PassWrapper<SdyRoundTripShardMapExportPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdyRoundTripShardMapExportPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    MLIRContext* context = moduleOp.getContext();
    auto builder = mlir::OpBuilder(context);
    mlir::SymbolTableCollection symbolTableCollection;
    mlir::SymbolTable& symbolTable =
        symbolTableCollection.getSymbolTable(moduleOp);
    moduleOp->walk([&](sdy::ManualComputationOp manualComputation) {
      builder.setInsertionPoint(moduleOp);
      auto funcOp = builder.create<FuncOp>(
          manualComputation.getLoc(), "shmap_body",
          builder.getFunctionType(
              manualComputation.getBody().getArgumentTypes(),
              sdy::getBodyTerminatorOpOperandTypes(manualComputation)));
      sdy::inlineRegionAndConvertTerminatorOp<mlir::func::ReturnOp>(
          manualComputation.getBody(), funcOp.getBody());
      symbolTable.insert(funcOp);

      builder.setInsertionPoint(manualComputation);
      auto customCallOp = builder.create<stablehlo::CustomCallOp>(
          manualComputation.getLoc(), manualComputation.getResultTypes(),
          manualComputation->getOperands(),
          mlir::ArrayRef<NamedAttribute>{
              builder.getNamedAttr("call_target_name",
                                   builder.getStringAttr(
                                       kManualComputationCustomCallTargetName)),
              builder.getNamedAttr(
                  "called_computations",
                  mlir::ArrayAttr::get(
                      context,
                      mlir::FlatSymbolRefAttr::get(funcOp.getSymNameAttr())))});
      addFrontendAttribute(customCallOp, kInShardings,
                           manualComputation.getInShardings());
      addFrontendAttribute(customCallOp, kOutShardings,
                           manualComputation.getOutShardings());
      addFrontendAttribute(customCallOp, kManualAxes,
                           manualComputation.getManualAxesAttr());

      manualComputation.replaceAllUsesWith(customCallOp->getResults());
      manualComputation.erase();
    });
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-shard-map-export";
  }

  StringRef getDescription() const override {
    return "Converts the body of a ManualComputationOps to a separate function "
           "with a CustomCallOp of the same name referring to it. The "
           "CustomCallOp saves the in/out shardings and manual axes as "
           "frontend attrs for HLO round tripping.";
  }
  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<sdy::SdyDialect, mlir::stablehlo::StablehloDialect>();
  }
};

}  // namespace

void registerSdyRoundTripShardMapExportPass() {
  mlir::registerPass(createSdyRoundTripShardMapExportPass);
}

std::unique_ptr<mlir::Pass> createSdyRoundTripShardMapExportPass() {
  return std::make_unique<SdyRoundTripShardMapExportPass>();
}

}  // namespace sdy
}  // namespace xla
