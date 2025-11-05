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

#include "xla/backends/gpu/codegen/triton/collective_emitter.h"

#include <memory>
#include <optional>
#include <ostream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/codegen/triton/fusion.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {
using ::tsl::proto_testing::EqualsProto;

class CollectiveBlockLevelConfigTest : public HloHardwareIndependentTestBase {
 public:
  CollectiveBlockLevelConfigTest()
      : device_info_{TestGpuDeviceInfo::RTXH100SXMDeviceInfo()} {}

  void Init(const Shape& shape) {
    const std::string module_str = GetModuleStr(shape);
    VLOG(0) << "module_str: " << module_str;
    TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(module_str));
    const HloInstruction* instr =
        module_->entry_computation()->GetInstructionWithName(
            "all-reduce-start");
    module_with_fusion_ =
        NewModuleWithFusion(instr, HloInstruction::FusionKind::kLoop);
    module_with_fusion_->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_unsupported_use_all_reduce_one_shot_kernel(true);
    fusion_instr_ = Cast<HloFusionInstruction>(
        module_with_fusion_->entry_computation()->root_instruction());
  }

 protected:
  static std::string GetModuleStr(const Shape& shape) {
    return absl::StrFormat(R"(
      HloModule test
      apply_op {
        x = f32[] parameter(0)
        y = f32[] parameter(1)
        ROOT apply_op = f32[] add(x, y)
      }

      ENTRY test_computation {
        param_0 = %1$s parameter(0)
        all-reduce-start = %1$s all-reduce-start(param_0), to_apply=apply_op, replica_groups={{0,1}}
        ROOT all-reduce-done = %1$s all-reduce-done(all-reduce-start)
      }
    )",
                           shape.ToString());
  }

  se::DeviceDescription device_info_;
  std::unique_ptr<HloModule> module_;
  std::unique_ptr<HloModule> module_with_fusion_;
  HloFusionInstruction* fusion_instr_;
};

class CollectiveEmitterTest : public CollectiveBlockLevelConfigTest {
 public:
  void Init(const Shape& shape) {
    CollectiveBlockLevelConfigTest::Init(shape);
    TF_ASSERT_OK_AND_ASSIGN(
        bool collective_fusion_config_set,
        TrySetGpuBackendConfigForCollective(device_info_, fusion_instr_));
    ASSERT_TRUE(collective_fusion_config_set);

    analysis_ = HloFusionAnalysis::Create(*fusion_instr_, device_info_);
    mlir_context_ = std::make_unique<mlir::MLIRContext>();
    symbolic_expr_context_ =
        std::make_unique<SymbolicExprContext>(mlir_context_.get());
    emitter_ = GetFusionEmitter(PreBufferAssignmentFusionInfo{*analysis_},
                                symbolic_expr_context_.get());
    ASSERT_NE(dynamic_cast<TritonFusion*>(emitter_.get()), nullptr);
  }

  TritonFusion* MutableEmitter() {
    CHECK_NE(emitter_.get(), nullptr);
    return dynamic_cast<TritonFusion*>(emitter_.get());
  }

  const TritonFusion* Emitter() const {
    CHECK_NE(emitter_.get(), nullptr);
    return dynamic_cast<const TritonFusion*>(emitter_.get());
  }

 protected:
  std::unique_ptr<mlir::MLIRContext> mlir_context_;
  std::unique_ptr<SymbolicExprContext> symbolic_expr_context_;
  std::optional<HloFusionAnalysis> analysis_;
  std::unique_ptr<FusionInterface> emitter_;
};

struct AllReduceBlockLevelConfigTestCase {
  std::string test_name;
  Shape shape;
  std::string expected_proto;

  // Teach gTest how to print the test case.
  [[maybe_unused]] friend void PrintTo(
      const AllReduceBlockLevelConfigTestCase& test_case, std::ostream* os) {
    *os << "{test_name: " << test_case.test_name
        << " shape: " << test_case.shape.ToString()
        << " expected_proto: " << test_case.expected_proto << "}";
  }
};

class CollectiveEmitterParameterizedTest
    : public CollectiveBlockLevelConfigTest,
      public ::testing::WithParamInterface<AllReduceBlockLevelConfigTestCase> {
};

TEST_P(CollectiveEmitterParameterizedTest, AllReduceBlockLevelConfig) {
  const auto& param = GetParam();
  Init(param.shape);
  TF_ASSERT_OK_AND_ASSIGN(
      const auto block_level_config,
      GetCollectiveBlockLevelFusionConfig(device_info_, fusion_instr_));
  ASSERT_TRUE(block_level_config.has_value());
  EXPECT_THAT(*block_level_config, EqualsProto(param.expected_proto));
}

INSTANTIATE_TEST_SUITE_P(
    CollectiveEmitterParameterizedTestInstantiation,
    CollectiveEmitterParameterizedTest,
    ::testing::Values(AllReduceBlockLevelConfigTestCase{
                          /* .test_name = */ "F32_65536",
                          /* .shape = */ ShapeUtil::MakeShape(F32, {65536}),
                          /* .expected_proto = */ R"pb(
                            num_warps: 16
                            num_ctas: 1
                            num_stages: 1
                            output_tiles { sizes: 2048 }
                          )pb"},
                      AllReduceBlockLevelConfigTestCase{
                          /* .test_name= */ "F32_200_100",
                          /* .shape= */ ShapeUtil::MakeShape(F32, {200, 100}),
                          /* .expected_proto= */ R"pb(
                            num_warps: 16
                            num_ctas: 1
                            num_stages: 1
                            output_tiles { sizes: 256 sizes: 8 }
                          )pb"}),
    [](const ::testing::TestParamInfo<
        CollectiveEmitterParameterizedTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST_F(CollectiveEmitterTest, AllReduceWithTritonGetLaunchConfig) {
  Init(ShapeUtil::MakeShape(F32, {65536}));
  const TritonFusion* triton_fusion = Emitter();
  ASSERT_NE(triton_fusion, nullptr);
  auto const launch_config = triton_fusion->GetLaunchConfig();
  ASSERT_NE(launch_config, std::nullopt);
  EXPECT_EQ(launch_config->launch_dimensions.num_blocks(), 32);
  EXPECT_EQ(launch_config->launch_dimensions.num_threads_per_block(), 512);
}

}  // namespace

}  // namespace xla::gpu
