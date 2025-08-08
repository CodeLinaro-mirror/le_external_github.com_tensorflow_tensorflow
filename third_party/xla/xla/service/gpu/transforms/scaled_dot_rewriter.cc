/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/scaled_dot_rewriter.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

bool IsUpscaleType(PrimitiveType type) {
  return type == PrimitiveType::F8E4M3FN || type == PrimitiveType::F8E5M2 ||
         type == PrimitiveType::S4 || type == PrimitiveType::F4E2M1FN ||
         type == PrimitiveType::F8E8M0FNU;
}

HloInstruction* UpscaleType(HloInstruction* instr) {
  if (!IsUpscaleType(instr->shape().element_type())) {
    return instr;
  }
  auto new_shape = instr->shape();
  new_shape.set_element_type(PrimitiveType::BF16);
  return instr->parent()->AddInstruction(
      HloInstruction::CreateConvert(new_shape, instr));
}

absl::Status CheckOperandAndScaleShapes(absl::string_view side,
                                        const HloInstruction* operand,
                                        const HloInstruction* scale) {
  if (operand->shape().dimensions().size() !=
      scale->shape().dimensions().size()) {
    return InvalidArgument(
        "%s: operand and scale must have the same rank: %d vs %d", side,
        operand->shape().dimensions().size(),
        scale->shape().dimensions().size());
  }

  for (int i = 0; i < operand->shape().dimensions().size(); ++i) {
    if (operand->shape().dimensions(i) % scale->shape().dimensions(i)) {
      return InvalidArgument(
          "%s: operand and scale dimensions must match or scale dimension must "
          "be divider of operand dimension: %d vs %d at index %d",
          side, operand->shape().dimensions(i), scale->shape().dimensions(i),
          i);
    }
  }
  return absl::OkStatus();
}

HloInstruction* BroadcastAndReshape(HloInstruction* scale,
                                    const Shape& operand_shape,
                                    HloComputation* computation) {
  Shape scale_shape = scale->shape();
  std::vector<int64_t> broadcast_dims;
  std::vector<int64_t> shape_dims;
  const int last_dim_index = operand_shape.dimensions().size() - 1;
  for (int i = 0; i < operand_shape.dimensions().size(); ++i) {
    broadcast_dims.push_back(i);
    shape_dims.push_back(scale_shape.dimensions(i));
  }
  shape_dims.push_back(operand_shape.dimensions(last_dim_index) /
                       scale_shape.dimensions(last_dim_index));
  Shape new_scales_shape(scale_shape.element_type(), shape_dims);
  HloInstruction* new_scales = computation->AddInstruction(
      HloInstruction::CreateBroadcast(new_scales_shape, scale, broadcast_dims));
  Shape reshaped_scales_shape(scale_shape.element_type(),
                              operand_shape.dimensions());
  return computation->AddInstruction(
      HloInstruction::CreateReshape(reshaped_scales_shape, new_scales));
}

}  // namespace

absl::StatusOr<bool> ScaledDotRewriter::Run(
    HloModule* module, const absl::flat_hash_set<absl::string_view>&) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kScaledDot) {
        continue;
      }
      changed = true;
      HloScaledDotInstruction* dot = Cast<HloScaledDotInstruction>(instruction);

      HloInstruction* lhs = UpscaleType(dot->mutable_operand(0));
      HloInstruction* lhs_scale = UpscaleType(dot->mutable_operand(1));
      HloInstruction* rhs = UpscaleType(dot->mutable_operand(2));
      HloInstruction* rhs_scale = UpscaleType(dot->mutable_operand(3));

      TF_RETURN_IF_ERROR(CheckOperandAndScaleShapes("LHS", lhs, lhs_scale));
      TF_RETURN_IF_ERROR(CheckOperandAndScaleShapes("RHS", rhs, rhs_scale));

      HloInstruction* lhs_scale_bc =
          BroadcastAndReshape(lhs_scale, lhs->shape(), computation);
      HloInstruction* rhs_scale_bc =
          BroadcastAndReshape(rhs_scale, rhs->shape(), computation);

      HloInstruction* lhs_dq =
          computation->AddInstruction(HloInstruction::CreateBinary(
              lhs->shape(), HloOpcode::kMultiply, lhs, lhs_scale_bc));
      HloInstruction* rhs_dq =
          computation->AddInstruction(HloInstruction::CreateBinary(
              rhs->shape(), HloOpcode::kMultiply, rhs, rhs_scale_bc));

      TF_RETURN_IF_ERROR(dot->ReplaceAllUsesWith(
          computation->AddInstruction(HloInstruction::CreateDot(
              dot->shape(), lhs_dq, rhs_dq, dot->dot_dimension_numbers(),
              dot->precision_config()))));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(dot));
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
