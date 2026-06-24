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

#include "xla/hlo/analysis/logical_buffer_analysis.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

class LogicalBufferAnalysisTest : public HloHardwareIndependentTestBase {
 protected:
  std::unique_ptr<LogicalBufferAnalysis> analysis_;

  // Verifies that a buffer is defined by `instruction` at `index` and matches.
  void VerifyBufferDefinedAt(HloInstruction* instruction,
                             const ShapeIndex& index) {
    auto buffer_or = analysis_->GetBuffer(instruction, index);
    ASSERT_OK(buffer_or.status());
    const LogicalBuffer& buffer = *buffer_or.value();
    EXPECT_EQ(buffer.instruction(), instruction);
    EXPECT_EQ(buffer.index(), index);
    auto buffer_by_id_or = analysis_->GetBuffer(buffer.id());
    ASSERT_OK(buffer_by_id_or.status());
    EXPECT_EQ(buffer_by_id_or.value(), &buffer);
  }

  // Verifies that no buffer is defined by `instruction` at `index`.
  void VerifyNoBufferDefinedAt(const HloInstruction* instruction,
                               const ShapeIndex& index) {
    for (const auto& buf : analysis_->logical_buffers()) {
      if (buf->instruction() == instruction && buf->index() == index) {
        ADD_FAILURE() << "Instruction " << instruction->name() << " at index "
                      << index.ToString()
                      << " should not define a logical buffer.";
      }
    }
  }

  // Returns all defining locations present in the analysis.
  std::vector<std::pair<const HloInstruction*, ShapeIndex>> GetDefiningSites() {
    std::vector<std::pair<const HloInstruction*, ShapeIndex>> defining_sites;
    for (const auto& buf : analysis_->logical_buffers()) {
      defining_sites.push_back({buf->instruction(), buf->index()});
    }
    return defining_sites;
  }
};

TEST_F(LogicalBufferAnalysisTest, InvalidGetBufferThrows) {
  const absl::string_view hlo_str = R"(
  HloModule module

  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    ROOT const = f32[] constant(1.0)
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
  ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(module.get()));

  HloInstruction* param = FindInstruction(module.get(), "p0");

  EXPECT_FALSE(analysis_->GetBuffer(param, {0}).ok());
  EXPECT_EQ(analysis_->GetBuffer(param, {0}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(LogicalBufferAnalysisTest, InvalidGetBufferIdBehavior) {
  const absl::string_view hlo_str = R"(
  HloModule module

  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    ROOT const = f32[] constant(1.0)
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
  ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(module.get()));

  // GetBuffer with invalid ID of 100 on a module containing 2 buffers.
  // In debug mode, this should trigger an out-of-bounds assertion/death.
  EXPECT_FALSE(analysis_->GetBuffer(100).ok());
  EXPECT_EQ(analysis_->GetBuffer(100).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace xla
