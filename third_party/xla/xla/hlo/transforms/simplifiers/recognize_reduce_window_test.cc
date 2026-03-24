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

#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace {

class RecognizeReduceWindowTest : public HloHardwareIndependentTestBase {
 public:
  void CheckRecognizeReduceWindow(absl::string_view hlo,
                                  std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, RecognizeReduceWindow{}, expected);
  }
};

TEST_F(RecognizeReduceWindowTest, AddTwoSlices) {
  const absl::string_view hlo_string = R"(
HloModule AddTwoSlices

ENTRY main {
  x = f32[10] parameter(0)
  slice_1 = f32[8] slice(x), slice={[0:8]}
  slice_2 = f32[8] slice(x), slice={[2:10]}
  ROOT add = f32[8] add(slice_1, slice_2)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: %[[REDUCER_NAME:.*]] ({{.*}}) -> f32[] {
// CHECK:   [[LHS:%.*]] = f32[] parameter(0)
// CHECK:   [[RHS:%.*]] = f32[] parameter(1)
// CHECK:   ROOT {{.*}} = f32[] add([[LHS]], [[RHS]])
// CHECK: }
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[8] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE:%.*]] = f32[10]{0} slice([[X]]), slice={[0:10]}
// CHECK:   [[ZERO:%.*]] = f32[] constant(0)
// CHECK:   ROOT {{.*}} = f32[8]{0} reduce-window([[SLICE]], [[ZERO]]), window={size=2 rhs_dilate=2}, to_apply=%[[REDUCER_NAME]]
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, AddTwoPads) {
  const absl::string_view hlo_string = R"(
HloModule AddTwoPads

ENTRY main {
  x = f32[10] parameter(0)
  pad_val = f32[] constant(0.0)
  pad_1 = f32[12] pad(x, pad_val), padding=2_0
  pad_2 = f32[12] pad(x, pad_val), padding=0_2
  ROOT add = f32[12] add(pad_1, pad_2)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: %[[REDUCER_NAME:.*]] ({{.*}}) -> f32[] {
// CHECK:   [[LHS:%.*]] = f32[] parameter(0)
// CHECK:   [[RHS:%.*]] = f32[] parameter(1)
// CHECK:   ROOT {{.*}} = f32[] add([[LHS]], [[RHS]])
// CHECK: }
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[12] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[PAD_VAL:%.*]] = f32[] constant(0)
// CHECK:   [[PAD_UNION:%.*]] = f32[14]{0} pad([[X]], [[PAD_VAL]]), padding=2_2
// CHECK:   [[ZERO:%.*]] = f32[] constant(0)
// CHECK:   ROOT {{.*}} = f32[12]{0} reduce-window([[PAD_UNION]], [[ZERO]]), window={size=2 rhs_dilate=2}, to_apply=%[[REDUCER_NAME]]
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, AddChainedSlices) {
  const absl::string_view hlo_string = R"(
HloModule AddChainedSlices

ENTRY main {
  x = f32[10] parameter(0)
  slice_1 = f32[6] slice(x), slice={[0:6]}
  slice_2 = f32[6] slice(x), slice={[2:8]}
  slice_3 = f32[6] slice(x), slice={[4:10]}
  add_1 = f32[6] add(slice_1, slice_2)
  ROOT add_2 = f32[6] add(add_1, slice_3)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: %[[REDUCER_NAME:.*]] ({{.*}}) -> f32[] {
// CHECK:   [[LHS:%.*]] = f32[] parameter(0)
// CHECK:   [[RHS:%.*]] = f32[] parameter(1)
// CHECK:   ROOT {{.*}} = f32[] add([[LHS]], [[RHS]])
// CHECK: }
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[6] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE:%.*]] = f32[10]{0} slice([[X]]), slice={[0:10]}
// CHECK:   [[ZERO:%.*]] = f32[] constant(0)
// CHECK:   ROOT {{.*}} = f32[6]{0} reduce-window([[SLICE]], [[ZERO]]), window={size=3 rhs_dilate=2}, to_apply=%[[REDUCER_NAME]]
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, SubtractSlicesToDotGeneral) {
  const absl::string_view hlo_string = R"(
HloModule SubtractSlices

ENTRY main {
  x = f32[10] parameter(0)
  slice_1 = f32[8] slice(x), slice={[0:8]}
  slice_2 = f32[8] slice(x), slice={[2:10]}
  ROOT sub = f32[8] subtract(slice_1, slice_2)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[8] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE_1:%.*]] = f32[8]{0} slice([[X]]), slice={[0:8]}
// CHECK:   [[RESHAPE:%.*]] = f32[1,8]{1,0} reshape([[SLICE_1]])
// CHECK:   [[SLICE_2:%.*]] = f32[8]{0} slice([[X]]), slice={[2:10]}
// CHECK:   [[RESHAPE_1:%.*]] = f32[1,8]{1,0} reshape([[SLICE_2]])
// CHECK:   [[CONCAT:%.*]] = f32[2,8]{1,0} concatenate([[RESHAPE]], [[RESHAPE_1]]), dimensions={0}
// CHECK:   [[CONSTANT:%.*]] = f32[2]{0} constant({1, -1})
// CHECK:   ROOT {{.*}} = f32[8]{0} dot([[CONCAT]], [[CONSTANT]]), lhs_contracting_dims={0}, rhs_contracting_dims={0}
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, FlattenAndSortWeightedSlices) {
  const absl::string_view hlo_string = R"(
HloModule FlattenSlices

ENTRY main {
  x = f32[10] parameter(0)
  slice_0 = f32[8] slice(x), slice={[1:9]}
  slice_1 = f32[8] slice(x), slice={[0:8]}
  slice_2 = f32[8] slice(x), slice={[2:10]}
  sub_1 = f32[8] subtract(slice_1, slice_2)
  ROOT add = f32[8] add(sub_1, slice_0)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[8] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE_1:%.*]] = f32[8]{0} slice([[X]]), slice={[0:8]}
// CHECK:   [[RESHAPE_1:%.*]] = f32[1,8]{1,0} reshape([[SLICE_1]])
// CHECK:   [[SLICE_0:%.*]] = f32[8]{0} slice([[X]]), slice={[1:9]}
// CHECK:   [[RESHAPE_0:%.*]] = f32[1,8]{1,0} reshape([[SLICE_0]])
// CHECK:   [[SLICE_2:%.*]] = f32[8]{0} slice([[X]]), slice={[2:10]}
// CHECK:   [[RESHAPE_2:%.*]] = f32[1,8]{1,0} reshape([[SLICE_2]])
// CHECK:   [[CONCAT:%.*]] = f32[3,8]{1,0} concatenate([[RESHAPE_1]], [[RESHAPE_0]], [[RESHAPE_2]]), dimensions={0}
// CHECK:   [[CONSTANT:%.*]] = f32[3]{0} constant({1, 1, -1})
// CHECK:   ROOT {{.*}} = f32[8]{0} dot([[CONCAT]], [[CONSTANT]]), lhs_contracting_dims={0}, rhs_contracting_dims={0}
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, WeightedSlicesToDotGeneral) {
  const absl::string_view hlo_string = R"(
HloModule WeightedSlices

ENTRY main {
  x = f32[10] parameter(0)
  slice_1 = f32[8] slice(x), slice={[0:8]}
  c1 = f32[] constant(3)
  b1 = f32[8] broadcast(c1), dimensions={}
  m1 = f32[8] multiply(slice_1, b1)
  slice_2 = f32[8] slice(x), slice={[2:10]}
  c2 = f32[] constant(4)
  b2 = f32[8] broadcast(c2), dimensions={}
  m2 = f32[8] multiply(slice_2, b2)
  ROOT add = f32[8] add(m1, m2)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[8] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE_1:%.*]] = f32[8]{0} slice([[X]]), slice={[0:8]}
// CHECK:   [[RESHAPE_1:%.*]] = f32[1,8]{1,0} reshape([[SLICE_1]])
// CHECK:   [[SLICE_2:%.*]] = f32[8]{0} slice([[X]]), slice={[2:10]}
// CHECK:   [[RESHAPE_2:%.*]] = f32[1,8]{1,0} reshape([[SLICE_2]])
// CHECK:   [[CONCAT:%.*]] = f32[2,8]{1,0} concatenate([[RESHAPE_1]], [[RESHAPE_2]]), dimensions={0}
// CHECK:   [[CONSTANT:%.*]] = f32[2]{0} constant({3, 4})
// CHECK:   ROOT {{.*}} = f32[8]{0} dot([[CONCAT]], [[CONSTANT]]), lhs_contracting_dims={0}, rhs_contracting_dims={0}
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, NonMatchingSlices) {
  const absl::string_view hlo_string = R"(
HloModule NonMatchingSlices

ENTRY main {
  x = f32[10] parameter(0)
  y = f32[10] parameter(1)
  slice_1 = f32[8] slice(x), slice={[0:8]}
  slice_2 = f32[8] slice(y), slice={[2:10]}
  ROOT add = f32[8] add(slice_1, slice_2)
}
)";
  // Should not match because slices come from different base parameters.
  CheckRecognizeReduceWindow(hlo_string, std::nullopt);
}

TEST_F(RecognizeReduceWindowTest, NonMatchingOpcode) {
  const absl::string_view hlo_string = R"(
HloModule NonMatchingOpcode

ENTRY main {
  x = f32[10] parameter(0)
  slice_1 = f32[8] slice(x), slice={[0:8]}
  slice_2 = f32[8] slice(x), slice={[2:10]}
  ROOT div = f32[8] divide(slice_1, slice_2)
}
)";
  // Should not match because divide is not commutative.
  CheckRecognizeReduceWindow(hlo_string, std::nullopt);
}

}  // namespace
}  // namespace xla
