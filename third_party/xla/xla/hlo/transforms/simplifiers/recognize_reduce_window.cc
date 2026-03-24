/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/recognize_reduce_window.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/window_util.h"

namespace xla {
namespace {

bool IsCommutative(const HloInstruction* inst) {
  return HloPredicateIsOp<HloOpcode::kAdd, HloOpcode::kMultiply,
                          HloOpcode::kMaximum, HloOpcode::kMinimum,
                          HloOpcode::kAnd, HloOpcode::kOr, HloOpcode::kXor>(
      inst);
}

HloComputation* GetOrCreateReducer(HloModule* module, HloOpcode opcode,
                                   PrimitiveType type) {
  HloComputation::Builder b("recognize_rw_reducer");
  auto p0 = b.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(type, {}), "lhs"));
  auto p1 = b.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(type, {}), "rhs"));
  b.AddInstruction(HloInstruction::CreateBinary(ShapeUtil::MakeShape(type, {}),
                                                opcode, p0, p1));
  return module->AddEmbeddedComputation(b.Build());
}

HloInstruction* CreateInitValue(HloComputation* comp, HloOpcode opcode,
                                PrimitiveType type) {
  switch (opcode) {
    case HloOpcode::kAdd:
      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(type)));
    case HloOpcode::kMultiply:
      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::One(type)));
    case HloOpcode::kMaximum:
      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::MinValue(type)));
    case HloOpcode::kMinimum:
      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::MaxValue(type)));
    case HloOpcode::kAnd:
      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::One(type)));
    case HloOpcode::kOr:
      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(type)));
    case HloOpcode::kXor:
      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(type)));
    default:
      return nullptr;
  }
}

struct WeightedSlice {
  HloInstruction* slice;
  double weight;
};

template <typename T>
Literal CreateWeightsLiteral(const std::vector<WeightedSlice>& folded) {
  std::vector<T> weights;
  weights.reserve(folded.size());
  for (const auto& ws : folded) {
    weights.push_back(static_cast<T>(ws.weight));
  }
  return LiteralUtil::CreateR1<T>(weights);
}

bool ExtractWeightedSlices(HloInstruction* inst,
                           std::vector<WeightedSlice>& slices,
                           double current_weight = 1.0) {
  if (inst->opcode() == HloOpcode::kSlice) {
    slices.push_back({inst, current_weight});
    return true;
  }
  if (inst->opcode() == HloOpcode::kMultiply) {
    HloInstruction* lhs = inst->mutable_operand(0);
    HloInstruction* rhs = inst->mutable_operand(1);
    std::optional<double> val = std::nullopt;
    HloInstruction* other = nullptr;

    auto get_const = [](HloInstruction* op) -> std::optional<double> {
      if (op->opcode() == HloOpcode::kBroadcast) {
        op = op->mutable_operand(0);
      }
      if (op->opcode() == HloOpcode::kConstant) {
        if (ShapeUtil::IsScalar(op->shape())) {
          return op->literal().GetAsDouble({});
        }
        if (op->literal().IsAllFirst()) {
          std::vector<int64_t> zeros(op->shape().dimensions_size(), 0);
          return op->literal().GetAsDouble(zeros);
        }
      }
      return std::nullopt;
    };

    if (auto l_val = get_const(lhs)) {
      val = *l_val;
      other = rhs;
    } else if (auto r_val = get_const(rhs)) {
      val = *r_val;
      other = lhs;
    }

    if (val.has_value() && other != nullptr) {
      return ExtractWeightedSlices(other, slices, current_weight * (*val));
    }
  }
  if (inst->opcode() == HloOpcode::kAdd) {
    return ExtractWeightedSlices(inst->mutable_operand(0), slices,
                                 current_weight) &&
           ExtractWeightedSlices(inst->mutable_operand(1), slices,
                                 current_weight);
  }
  if (inst->opcode() == HloOpcode::kSubtract) {
    return ExtractWeightedSlices(inst->mutable_operand(0), slices,
                                 current_weight) &&
           ExtractWeightedSlices(inst->mutable_operand(1), slices,
                                 -current_weight);
  }
  if (inst->opcode() == HloOpcode::kDot) {
    HloInstruction* lhs = inst->mutable_operand(0);
    HloInstruction* rhs = inst->mutable_operand(1);

    if (lhs->opcode() == HloOpcode::kConcatenate &&
        rhs->opcode() == HloOpcode::kConstant &&
        inst->dot_dimension_numbers().lhs_contracting_dimensions_size() == 1 &&
        inst->dot_dimension_numbers().lhs_contracting_dimensions(0) == 0 &&
        inst->dot_dimension_numbers().rhs_contracting_dimensions_size() == 1 &&
        inst->dot_dimension_numbers().rhs_contracting_dimensions(0) == 0 &&
        lhs->shape().dimensions_size() > 0 &&
        rhs->shape().dimensions_size() == 1 &&
        lhs->shape().dimensions(0) == rhs->shape().dimensions(0)) {
      int64_t num_slices = lhs->operand_count();
      bool all_good = true;
      std::vector<WeightedSlice> temp_slices;
      for (int64_t i = 0; i < num_slices; ++i) {
        HloInstruction* reshape = lhs->mutable_operand(i);
        if (reshape->opcode() != HloOpcode::kReshape ||
            reshape->operand(0)->opcode() != HloOpcode::kSlice) {
          all_good = false;
          break;
        }
        std::optional<double> w = rhs->literal().GetAsDouble({i});
        if (!w.has_value()) {
          all_good = false;
          break;
        }
        temp_slices.push_back(
            {reshape->mutable_operand(0), current_weight * (*w)});
      }
      if (all_good) {
        slices.insert(slices.end(), temp_slices.begin(), temp_slices.end());
        return true;
      }
    }
  }
  return false;
}

absl::StatusOr<bool> RunOnComputation(HloComputation* computation) {
  bool changed = false;
  std::vector<HloInstruction*> instructions =
      computation->MakeInstructionPostOrder();

  for (HloInstruction* inst : instructions) {
    if (inst->IsDead()) {
      continue;
    }
    if (inst->operand_count() != 2) {
      continue;
    }
    if (!IsCommutative(inst) && inst->opcode() != HloOpcode::kSubtract) {
      continue;
    }

    HloInstruction* lhs = inst->mutable_operand(0);
    HloInstruction* rhs = inst->mutable_operand(1);

    if (inst->opcode() == HloOpcode::kSubtract ||
        inst->opcode() == HloOpcode::kAdd) {
      std::vector<WeightedSlice> lhs_slices, rhs_slices;
      // For subtract/add with weighted slice combinations we build dot
      // convolution. But we require matching both sides initially to avoid
      // turning single raw slices into dots unnecessarily.
      if (ExtractWeightedSlices(lhs, lhs_slices, 1.0) &&
          ExtractWeightedSlices(
              rhs, rhs_slices,
              inst->opcode() == HloOpcode::kSubtract ? -1.0 : 1.0)) {
        std::vector<WeightedSlice> combined = lhs_slices;
        combined.insert(combined.end(), rhs_slices.begin(), rhs_slices.end());

        bool valid = !combined.empty();
        HloInstruction* base = nullptr;
        Shape slice_shape;
        int rank = -1;

        if (valid) {
          base = combined[0].slice->mutable_operand(0);
          slice_shape = combined[0].slice->shape();
          rank = slice_shape.dimensions_size();
          for (auto& ws : combined) {
            if (ws.slice->mutable_operand(0) != base ||
                !ShapeUtil::SameDimensions(ws.slice->shape(), slice_shape)) {
              valid = false;
              break;
            }
            if (!absl::c_all_of(ws.slice->slice_strides(),
                                [](int64_t s) { return s == 1; })) {
              valid = false;
              break;
            }
          }
        }

        int64_t differing_dim = -1;
        if (valid) {
          for (int i = 0; i < rank; ++i) {
            int64_t start = combined[0].slice->slice_starts(i);
            bool ok = true;
            for (size_t j = 1; j < combined.size(); ++j) {
              if (combined[j].slice->slice_starts(i) != start ||
                  combined[j].slice->slice_limits(i) !=
                      combined[0].slice->slice_limits(i)) {
                ok = false;
                break;
              }
            }
            if (!ok) {
              if (differing_dim != -1) {
                differing_dim = -1;
                valid = false;
                break;
              }
              differing_dim = i;
            }
          }
        }

        if (valid && differing_dim == -1) {
          differing_dim = 0;
        }

        if (valid) {
          std::sort(
              combined.begin(), combined.end(),
              [differing_dim](const WeightedSlice& a, const WeightedSlice& b) {
                return a.slice->slice_starts(differing_dim) <
                       b.slice->slice_starts(differing_dim);
              });

          std::vector<WeightedSlice> folded;
          for (auto& ws : combined) {
            if (folded.empty()) {
              folded.push_back(ws);
            } else {
              if (folded.back().slice->slice_starts(differing_dim) ==
                  ws.slice->slice_starts(differing_dim)) {
                folded.back().weight += ws.weight;
              } else {
                folded.push_back(ws);
              }
            }
          }

          folded.erase(std::remove_if(folded.begin(), folded.end(),
                                      [](const WeightedSlice& ws) {
                                        return std::abs(ws.weight) < 1e-6;
                                      }),
                       folded.end());

          // If we just mapped a vanilla commutative add over 2 slices, skip it
          // so the subsequent reduce_window pass fires instead!
          if (!folded.empty() &&
              !(inst->opcode() == HloOpcode::kAdd && folded.size() == 2 &&
                std::abs(folded[0].weight - 1.0) < 1e-6 &&
                std::abs(folded[1].weight - 1.0) < 1e-6)) {
            if (folded.size() == 1 && std::abs(folded[0].weight - 1.0) < 1e-6) {
              TF_CHECK_OK(
                  computation->ReplaceInstruction(inst, folded[0].slice));
              changed = true;
              continue;
            }

            Literal weights_literal;
            bool matched_type = true;
            switch (inst->shape().element_type()) {
              case F32:
                weights_literal = CreateWeightsLiteral<float>(folded);
                break;
              case F64:
                weights_literal = CreateWeightsLiteral<double>(folded);
                break;
              case BF16:
                weights_literal = CreateWeightsLiteral<bfloat16>(folded);
                break;
              case F16:
                weights_literal = CreateWeightsLiteral<half>(folded);
                break;
              case S32:
                weights_literal = CreateWeightsLiteral<int32_t>(folded);
                break;
              case S64:
                weights_literal = CreateWeightsLiteral<int64_t>(folded);
                break;
              default:
                matched_type = false;
            }

            if (matched_type && folded.size() >= 2) {
              Shape reshape_shape = slice_shape;
              std::vector<int64_t> concat_dims = {1};
              for (int64_t d : reshape_shape.dimensions())
                concat_dims.push_back(d);
              Shape new_shape = ShapeUtil::MakeShape(
                  reshape_shape.element_type(), concat_dims);

              std::vector<HloInstruction*> reshaped_slices;
              reshaped_slices.reserve(folded.size());
              for (auto& ws : folded) {
                reshaped_slices.push_back(computation->AddInstruction(
                    HloInstruction::CreateReshape(new_shape, ws.slice)));
              }

              Shape concat_shape = new_shape;
              concat_shape.set_dimensions(0, folded.size());
              HloInstruction* concat =
                  computation->AddInstruction(HloInstruction::CreateConcatenate(
                      concat_shape, reshaped_slices, 0));

              HloInstruction* weights = computation->AddInstruction(
                  HloInstruction::CreateConstant(std::move(weights_literal)));
              DotDimensionNumbers dnums;
              dnums.add_lhs_contracting_dimensions(0);
              dnums.add_rhs_contracting_dimensions(0);
              PrecisionConfig precision_config;
              precision_config.add_operand_precision(PrecisionConfig::DEFAULT);
              precision_config.add_operand_precision(PrecisionConfig::DEFAULT);

              HloInstruction* dot =
                  computation->AddInstruction(HloInstruction::CreateDot(
                      inst->shape(), concat, weights, dnums, precision_config));
              TF_CHECK_OK(computation->ReplaceInstruction(inst, dot));
              changed = true;
              continue;
            }
          }
        }
      }
    }

    int64_t dim = -1;
    int64_t window_dilation = -1;
    HloInstruction* new_base_op = nullptr;

    // Pattern 1: add(pad, pad)
    if (lhs->opcode() == HloOpcode::kPad && rhs->opcode() == HloOpcode::kPad) {
      if (lhs->operand(0) == rhs->operand(0) &&
          lhs->operand(1)->Identical(*rhs->operand(1))) {  // Same padding value
        const PaddingConfig& l_pad = lhs->padding_config();
        const PaddingConfig& r_pad = rhs->padding_config();

        bool matched = true;
        for (int i = 0; i < l_pad.dimensions_size(); ++i) {
          if (l_pad.dimensions(i).interior_padding() != 0 ||
              r_pad.dimensions(i).interior_padding() != 0) {
            matched = false;
            break;
          }
          int64_t l_low = l_pad.dimensions(i).edge_padding_low();
          int64_t l_high = l_pad.dimensions(i).edge_padding_high();
          int64_t r_low = r_pad.dimensions(i).edge_padding_low();
          int64_t r_high = r_pad.dimensions(i).edge_padding_high();

          if (l_low == r_low && l_high == r_high) {
            continue;  // Same padding in this dim
          }

          if (dim != -1) {
            matched = false;  // More than 1 dimension differs
            break;
          }

          if (l_low > 0 && l_low == r_high && l_high == 0 && r_low == 0) {
            dim = i;
            window_dilation = l_low;
          } else if (r_low > 0 && r_low == l_high && r_high == 0 &&
                     l_low == 0) {
            dim = i;
            window_dilation = r_low;
          } else {
            matched = false;
            break;
          }
        }

        if (matched && dim != -1) {
          PaddingConfig new_pad_config = l_pad;
          new_pad_config.mutable_dimensions(dim)->set_edge_padding_low(
              window_dilation);
          new_pad_config.mutable_dimensions(dim)->set_edge_padding_high(
              window_dilation);
          Shape pad_shape = lhs->shape();
          pad_shape.set_dimensions(
              dim,
              lhs->operand(0)->shape().dimensions(dim) + 2 * window_dilation);
          new_base_op = computation->AddInstruction(HloInstruction::CreatePad(
              pad_shape, lhs->mutable_operand(0), lhs->mutable_operand(1),
              new_pad_config));
        }
      }
    }

    // Pattern 2: add(slice, slice)
    if (!new_base_op && lhs->opcode() == HloOpcode::kSlice &&
        rhs->opcode() == HloOpcode::kSlice) {
      if (lhs->operand(0) == rhs->operand(0) &&
          absl::c_all_of(lhs->slice_strides(),
                         [](int64_t s) { return s == 1; }) &&
          absl::c_all_of(rhs->slice_strides(),
                         [](int64_t s) { return s == 1; })) {
        int rank = lhs->shape().dimensions_size();
        bool matched = true;
        std::vector<int64_t> new_starts(rank);
        std::vector<int64_t> new_limits(rank);

        for (int i = 0; i < rank; ++i) {
          if (lhs->slice_starts(i) == rhs->slice_starts(i) &&
              lhs->slice_limits(i) == rhs->slice_limits(i)) {
            new_starts[i] = lhs->slice_starts(i);
            new_limits[i] = lhs->slice_limits(i);
          } else if (dim != -1) {
            matched = false;
            break;
          } else {
            int64_t l_start = lhs->slice_starts(i);
            int64_t r_start = rhs->slice_starts(i);
            int64_t l_limit = lhs->slice_limits(i);
            int64_t r_limit = rhs->slice_limits(i);
            dim = i;
            new_starts[i] = std::min(l_start, r_start);
            new_limits[i] = std::max(l_limit, r_limit);
            window_dilation = std::abs(l_start - r_start);
            if (window_dilation == 0 ||
                (l_limit - l_start != r_limit - r_start)) {
              matched = false;
              break;
            }
          }
        }

        if (matched && dim != -1) {
          std::vector<int64_t> strides(rank, 1);
          Shape slice_shape = lhs->shape();
          slice_shape.set_dimensions(dim, new_limits[dim] - new_starts[dim]);
          new_base_op = computation->AddInstruction(
              HloInstruction::CreateSlice(slice_shape, lhs->mutable_operand(0),
                                          new_starts, new_limits, strides));
        }
      }
    }

    // Pattern 3: add(reduce_window, slice)
    int64_t current_window_size =
        2;  // For the first transformation, size becomes 2
    if (!new_base_op && (lhs->opcode() == HloOpcode::kReduceWindow ||
                         rhs->opcode() == HloOpcode::kReduceWindow)) {
      // Find which one is reduce_window and which is slice/pad
      HloInstruction* rw =
          (lhs->opcode() == HloOpcode::kReduceWindow) ? lhs : rhs;
      HloInstruction* other =
          (lhs->opcode() == HloOpcode::kReduceWindow) ? rhs : lhs;

      // Make sure the reducer is the same!
      if (rw->to_apply()->root_instruction()->opcode() == inst->opcode() &&
          other->opcode() == HloOpcode::kSlice &&
          rw->operand(0)->opcode() == HloOpcode::kSlice) {
        HloInstruction* rw_slice = rw->mutable_operand(0);
        if (rw_slice->operand(0) == other->operand(0) &&
            absl::c_all_of(rw_slice->slice_strides(),
                           [](int64_t s) { return s == 1; }) &&
            absl::c_all_of(other->slice_strides(),
                           [](int64_t s) { return s == 1; })) {
          int rank = rw_slice->shape().dimensions_size();
          bool matched = true;
          std::vector<int64_t> new_starts(rank);
          std::vector<int64_t> new_limits(rank);

          for (int i = 0; i < rank; ++i) {
            if (rw_slice->slice_starts(i) == other->slice_starts(i) &&
                rw_slice->slice_limits(i) == other->slice_limits(i)) {
              new_starts[i] = rw_slice->slice_starts(i);
              new_limits[i] = rw_slice->slice_limits(i);
            } else if (dim != -1) {
              matched = false;
              break;
            } else {
              dim = i;
              int64_t rw_start = rw_slice->slice_starts(i);
              int64_t o_start = other->slice_starts(i);
              int64_t rw_limit = rw_slice->slice_limits(i);
              int64_t o_limit = other->slice_limits(i);
              int64_t D = rw->window().dimensions(i).window_dilation();
              int64_t size = rw->window().dimensions(i).size();

              bool is_append =
                  (o_start == rw_start + D * size) && (o_limit == rw_limit + D);
              bool is_prepend =
                  (rw_start == o_start + D) && (rw_limit == o_limit + D * size);

              if (is_append || is_prepend) {
                window_dilation = D;
              } else {
                matched = false;
                break;
              }

              new_starts[i] = std::min(rw_start, o_start);
              new_limits[i] = std::max(rw_limit, o_limit);
              current_window_size = size + 1;
            }
          }
          if (matched && dim != -1) {
            std::vector<int64_t> strides(rank, 1);
            Shape slice_shape = rw_slice->shape();
            slice_shape.set_dimensions(dim, new_limits[dim] - new_starts[dim]);
            new_base_op =
                computation->AddInstruction(HloInstruction::CreateSlice(
                    slice_shape, rw_slice->mutable_operand(0), new_starts,
                    new_limits, strides));
          }
        }
      }
    }

    if (new_base_op) {
      HloComputation* reducer =
          inst->opcode() == ((lhs->opcode() == HloOpcode::kReduceWindow)
                                 ? lhs->to_apply()->root_instruction()->opcode()
                                 : inst->opcode())
              ? GetOrCreateReducer(computation->parent(), inst->opcode(),
                                   inst->shape().element_type())
              : nullptr;
      HloInstruction* init_val = CreateInitValue(computation, inst->opcode(),
                                                 inst->shape().element_type());
      if (!init_val || !reducer) {
        continue;
      }

      int rank = new_base_op->shape().dimensions_size();
      Window window = window_util::MakeWindow(std::vector<int64_t>(rank, 1));
      window.mutable_dimensions(dim)->set_size(current_window_size);
      window.mutable_dimensions(dim)->set_window_dilation(window_dilation);

      HloInstruction* rw =
          computation->AddInstruction(HloInstruction::CreateReduceWindow(
              inst->shape(), new_base_op, init_val, window, reducer));

      TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(rw));
      changed = true;
    }
  }

  return changed;
}

}  // namespace

absl::StatusOr<bool> RecognizeReduceWindow::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  std::vector<HloComputation*> computations =
      module->MakeNonfusionComputations(execution_threads);
  for (HloComputation* computation : computations) {
    TF_ASSIGN_OR_RETURN(bool computation_changed,
                        RunOnComputation(computation));
    changed |= computation_changed;
  }
  return changed;
}

}  // namespace xla
