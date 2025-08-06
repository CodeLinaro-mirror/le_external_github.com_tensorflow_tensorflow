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

#include "xla/service/debug/unstable_reduction_detector.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class UnstableReductionDetectorTest : public HloRunnerAgnosticTestBase {
 public:
  static constexpr absl::string_view kHlo = R"(
    red {
        p0 = bf16[] parameter(0)
        p1 = bf16[] parameter(1)
        ROOT red = bf16[] add(p0, p1)
    }

    ENTRY main {
        p0 = bf16[164] parameter(0)
        init = bf16[] constant(1.0)
        ROOT red = bf16[] reduce(p0, init), to_apply=red, dimensions={0}
    }
  )";
};

TEST(UnstableReductionDetectorTest, CrashOnUnstableReductions) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnUnverifiedModule(UnstableReductionDetectorTest::kHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_detect_unstable_reductions(
          DebugOptions::UNSTABLE_REDUCTION_DETECTION_MODE_CRASH);
  UnstableReductionDetector detector;
  EXPECT_DEATH(
      detector.Run(module.get(), /*execution_threads=*/{}).IgnoreError(),
      "Unstable reductions found in module '.*'");
}

TEST(UnstableReductionDetectorTest, WhenLogDetectOnUnstableReduction) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnUnverifiedModule(UnstableReductionDetectorTest::kHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_detect_unstable_reductions(
          DebugOptions::UNSTABLE_REDUCTION_DETECTION_MODE_LOG);
  UnstableReductionDetector detector;
  EXPECT_EQ(detector.CheckModule(module.get()),
            UnstableReductionDetector::Result::kUnstableReductionsFound);
}

TEST(UnstableReductionDetectorTest, WhenNoneDoNothing) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnUnverifiedModule(UnstableReductionDetectorTest::kHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_detect_unstable_reductions(
          DebugOptions::UNSTABLE_REDUCTION_DETECTION_MODE_NONE);
  UnstableReductionDetector detector;
  EXPECT_EQ(detector.CheckModule(module.get()),
            UnstableReductionDetector::Result::kSkipped);
}

}  // namespace
}  // namespace xla
