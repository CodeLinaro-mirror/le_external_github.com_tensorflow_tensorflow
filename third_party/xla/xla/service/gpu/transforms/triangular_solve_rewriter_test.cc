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

#include "xla/service/gpu/transforms/triangular_solve_rewriter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace m = ::xla::match;

namespace xla {
namespace gpu {
namespace {

using ::testing::status::IsOkAndHolds;

class TriangularSolveRewriterTest : public HloTestBase {};

TEST_F(TriangularSolveRewriterTest, SimpleTriangularSolve) {
  const char* const hlo_string = R"(
HloModule TriangularSolve

ENTRY main {
  a = f32[4,4]{1,0} parameter(0)
  b = f32[3,4]{1,0} parameter(1)
  ROOT triangular-solve = f32[3,4]{1,0} triangular-solve(a, b), lower=true,
                                          transpose_a=TRANSPOSE
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<TriangularSolveRewriter>();
  EXPECT_THAT(pipeline.Run(module.get()), IsOkAndHolds(true));

  /*
  ENTRY %main (a: f32[4,4], b: f32[3,4]) -> f32[3,4] {
    %a = f32[4,4]{1,0} parameter(0)
    %b = f32[3,4]{1,0} parameter(1)
    %triangular-solve.1 = (f32[3,4]{1,0}, s8[0]{0}) custom-call(%a, %b),
  custom_call_target="__cublas$triangularSolve",
  backend_config={"left_side":false,"lower":true,"unit_diagonal":false,"transpose_a":"TRANSPOSE"}
    ROOT %get-tuple-element = f32[3,4]{1,0}
  get-tuple-element(%triangular-solve.1), index=0
  }
  */

  HloInstruction* triangular_solve_custom_call;

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(m::CustomCall(
          &triangular_solve_custom_call, {"__cublas$triangularSolve"}))));
}

TEST_F(TriangularSolveRewriterTest, SimpleRightLowerNotranspose) {
  const char* const hlo_string = R"(
HloModule TriangularSolve_module, entry_computation_layout={(f32[4,4]{1,0}, f32[3,4]{1,0})->f32[3,4]{1,0}}

ENTRY %SimpleRightLowerNotranspose.4 (a.1: f32[4,4], b.2: f32[3,4]) -> f32[3,4] {
  %a.1 = f32[4,4]{1,0} parameter(0)
  %b.2 = f32[3,4]{1,0} parameter(1)
  ROOT %solve = f32[3,4]{1,0} triangular-solve(f32[4,4]{1,0} %a.1, f32[3,4]{1,0} %b.2), lower=true, transpose_a=NO_TRANSPOSE
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<TriangularSolveRewriter>();
  EXPECT_THAT(pipeline.Run(module.get()), IsOkAndHolds(true));

  /*
  ENTRY %SimpleRightLowerNotranspose.4 (a.1: f32[4,4], b.2: f32[3,4]) ->
  f32[3,4] { %a.1 = f32[4,4]{1,0} parameter(0) %b.2 = f32[3,4]{1,0} parameter(1)
  %triangular-solve = (f32[3,4]{1,0}, s8[0]{0}) custom-call(%a.1, %b.2),
  custom_call_target="__cublas$triangularSolve",
  backend_config={"left_side":false,"lower":true,"unit_diagonal":false,"transpose_a":"NO_TRANSPOSE"}
  ROOT %get-tuple-element = f32[3,4]{1,0} get-tuple-element(%triangular-solve),
  index=0
  }
  */
  HloInstruction* triangular_solve_custom_call;

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(m::CustomCall(
          &triangular_solve_custom_call, {"__cublas$triangularSolve"}))));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
