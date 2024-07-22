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

#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_attrs.h"

#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/gpu/model/indexing_map.h"

#define GET_ATTRDEF_LIST
#define GET_ATTRDEF_CLASSES
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_attrs.h.inc"

namespace xla {
namespace gpu {

using llvm::ParseResult;
using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::ArrayRef;
using mlir::AsmParser;
using mlir::AsmPrinter;
using mlir::failed;
using mlir::failure;
using mlir::success;

std::string IntervalToString(const Interval& interval) {
  return absl::StrFormat("[%d, %d]", interval.lower, interval.upper);
}

ParseResult ParseInterval(AsmParser& parser, Interval& interval) {
  return success(!(parser.parseLSquare() ||
                   parser.parseInteger(interval.lower) || parser.parseComma() ||
                   parser.parseInteger(interval.upper) ||
                   parser.parseRSquare()));
}

void PrintDimVars(AsmPrinter& p, ArrayRef<DimVar> dim_vars) {
  for (int i = 0; i < dim_vars.size(); ++i) {
    p << absl::StrFormat("d%d", i) << " in " << dim_vars[i].bounds << "\n";
  }
}

mlir::FailureOr<SmallVector<DimVar>> ParseDimVars(
    AsmParser& parser, SmallVector<std::string>& dim_strings) {
  SmallVector<DimVar> dim_vars;
  for (const auto& dim_string : dim_strings) {
    if (parser.parseKeyword(dim_string) || parser.parseKeyword("in") ||
        ParseInterval(parser, dim_vars.emplace_back().bounds)) {
      return failure();
    }
  }
  return dim_vars;
}

void PrintRangeVars(AsmPrinter& p, ArrayRef<RangeVar> range_vars) {
  for (int i = 0; i < range_vars.size(); ++i) {
    p << absl::StrFormat("s%d", i) << " in "
      << IntervalToString(range_vars[i].range) << "\n";
  }
}

mlir::FailureOr<SmallVector<RangeVar>> ParseRangeVars(
    AsmParser& parser, SmallVector<std::string>& range_strings) {
  SmallVector<RangeVar> range_vars;
  for (const auto& range_string : range_strings) {
    if (parser.parseKeyword(range_string) || parser.parseKeyword("in") ||
        ParseInterval(parser, range_vars.emplace_back().range)) {
      return failure();
    }
  }
  return range_vars;
}

void PrintConstraints(AsmPrinter& p,
                      ArrayRef<std::pair<AffineExpr, Interval>> constraints) {
  for (const auto& constraint : constraints) {
    p << constraint.first << " in " << IntervalToString(constraint.second)
      << "\n";
  }
}

mlir::FailureOr<SmallVector<std::pair<AffineExpr, Interval>>> ParseConstraints(
    AsmParser& parser,
    SmallVector<std::pair<llvm::StringRef, AffineExpr>> symbolSet) {
  SmallVector<std::pair<AffineExpr, Interval>> constraints;
  while (failed(parser.parseOptionalGreater())) {
    auto constraint = &constraints.emplace_back();
    if (parser.parseAffineExpr(symbolSet, constraint->first) ||
        parser.parseKeyword("in") ||
        ParseInterval(parser, constraint->second)) {
      return failure();
    }
  }
  return constraints;
}

mlir::Attribute IndexingMapAttr::parse(mlir::AsmParser& parser, mlir::Type) {
  mlir::AffineMap map;
  if (parser.parseLess() || parser.parseAffineMap(map)) {
    return {};
  }

  // Store real strings to back up StringRef throughout ParseConstraints.
  SmallVector<std::string> dim_strings(map.getNumDims());
  SmallVector<std::string> symbol_strings(map.getNumSymbols());
  SmallVector<std::pair<llvm::StringRef, AffineExpr>> symbolSet;
  symbolSet.reserve(map.getNumDims() + map.getNumSymbols());
  for (int i = 0; i < map.getNumDims(); ++i) {
    dim_strings[i] = absl::StrFormat("d%d", i);
    symbolSet.push_back(
        {dim_strings[i], mlir::getAffineDimExpr(i, parser.getContext())});
  }
  for (int i = 0; i < map.getNumSymbols(); ++i) {
    symbol_strings[i] = absl::StrFormat("s%d", i);
    symbolSet.push_back(
        {symbol_strings[i], mlir::getAffineSymbolExpr(i, parser.getContext())});
  }

  if (parser.parseKeyword("domain") || parser.parseColon()) {
    return {};
  }
  auto maybe_dim_vars = ParseDimVars(parser, dim_strings);
  if (failed(maybe_dim_vars)) {
    return {};
  }

  auto maybe_range_vars = ParseRangeVars(parser, symbol_strings);
  if (failed(maybe_range_vars)) {
    return {};
  }

  auto maybe_constraints = ParseConstraints(parser, symbolSet);
  if (failed(maybe_constraints)) {
    return {};
  }
  // ParseConstraints consumes the > to know when to stop.
  return IndexingMapAttr::get(parser.getContext(), map, *maybe_dim_vars,
                              *maybe_range_vars, *maybe_constraints);
}

void IndexingMapAttr::print(mlir::AsmPrinter& printer) const {
  printer << "<";
  printer.printStrippedAttrOrType(getMap());
  printer << "\ndomain:\n";
  PrintDimVars(printer, getDimVars());
  PrintRangeVars(printer, getRangeVars());
  PrintConstraints(printer, getConstraints());
  printer << ">";
}

}  // namespace gpu
}  // namespace xla
