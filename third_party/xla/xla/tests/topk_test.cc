/* Copyright 2023 The OpenXLA Authors.

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

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/xla_test_backend_predicates.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"

namespace xla {
namespace {

using TopkTest = HloPjRtInterpreterReferenceMixin<HloTestBase>;

TEST_F(TopkTest, LargestTopK) {
  absl::string_view hlo_text_module = R"(
    HloModule topk

    ENTRY TopK {
      x = bf16[10,10] parameter(0)
      ROOT topk = (bf16[10,2], s32[10,2]) topk(x), k=2, largest=true
    }
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text_module, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TopkTest, SmallestTopK) {
  absl::string_view hlo_text_module = R"(
    HloModule topk

    ENTRY TopK {
      x = bf16[10,10] parameter(0)
      ROOT topk = (bf16[10,2], s32[10,2]) topk(x), k=2, largest=false
    }
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text_module, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TopkTest, TopKOfTranspose) {
  // Regression test for b/362565176
  absl::string_view hlo_text_module = R"(
    HloModule topk

    ENTRY main {
      %Arg_0.1 = f32[2048,6]{1,0} parameter(0)
      t = f32[6,2048]{0,1} transpose(%Arg_0.1), dimensions={1,0}
      ROOT %topk.2 = (f32[6,8]{1,0}, s32[6,8]{1,0}) topk(t), k=8, largest=true
    }
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text_module, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TopkTest, TopKWithSpecialFloats) {
  // Regression test for TopK TotalOrder with special float values (NaN, +-Inf,
  // +-0) We use [8, 1024] since the __gpu$TopK kernel requires N >= 1024 to
  // trigger.
  float neg_qnan0 = absl::bit_cast<float>(0xFFC00000u);
  float neg_qnan1 = absl::bit_cast<float>(0xFFE00000u);
  float neg_qnan2 = absl::bit_cast<float>(0xFFE00001u);
  float pos_qnan0 = absl::bit_cast<float>(0x7FC00000u);
  float pos_qnan1 = absl::bit_cast<float>(0x7FE00000u);
  float pos_qnan2 = absl::bit_cast<float>(0x7FE00001u);

  std::vector<float> input_row_start = {
      -INFINITY, INFINITY, -1.0e-41f, 0.0f,     pos_qnan0,
      pos_qnan2, -0.0f,    -90.445f,  INFINITY, neg_qnan0,
      neg_qnan2, -90.445f, -0.0f,     1.0e-42f, -1.0e-43f,
      neg_qnan1, 0.0f,     -INFINITY, 90.445f,  pos_qnan1,
      0.0125f,   1.0e-40f, -1.0e-41f, 1.0e-42f, -1.0e-43f};

  // Pad out to 1024 to force exactly the TopK kernel custom call routing bypass
  Array2D<float> input(8, 1024, neg_qnan0);
  for (int i = 0; i < input_row_start.size(); ++i) {
    for (int j = 0; j < 8; ++j) {
      input(j, i) = input_row_start[i];
    }
  }

  auto literal = LiteralUtil::CreateR2FromArray2D<float>(input);

  for (int k : {2, 4, 8, 12, 16, 24, 32, 64}) {
    const auto hlo_text_module = absl::StrFormat(R"(
      HloModule topk

      ENTRY TopK {
        x = f32[8,1024]{1,0} parameter(0)
        ROOT topk = (f32[8,%d]{1,0}, s32[8,%d]{1,0}) topk(x), k=%d, largest=true
      }
    )",
                                                 k, k, k);

    ASSERT_OK_AND_ASSIGN(auto module,
                         ParseAndReturnVerifiedModule(hlo_text_module));
    // Interpreter matches TotalOrder logic, ensuring exact equivalence.
    EXPECT_TRUE(RunAndCompare(std::move(module),
                              absl::Span<const Literal* const>{&literal},
                              ErrorSpec{0, 0}));
  }
}

TEST_F(TopkTest, TopKWithSpecialBf16) {
  if (xla::test::DeviceTypeIs(xla::test::kTpu)) {
    GTEST_SKIP() << "Skip TopKWithSpecialBf16 on TPU.";
  }
  // Regression test for TopK TotalOrder with special bfloat16 values (NaN,
  // +-Inf, +-0)
  bfloat16 neg_qnan0 = absl::bit_cast<bfloat16>(static_cast<uint16_t>(0xFFC0u));
  bfloat16 neg_qnan1 = absl::bit_cast<bfloat16>(static_cast<uint16_t>(0xFFE0u));
  bfloat16 neg_qnan2 = absl::bit_cast<bfloat16>(static_cast<uint16_t>(0xFFE1u));
  bfloat16 pos_qnan0 = absl::bit_cast<bfloat16>(static_cast<uint16_t>(0x7FC0u));
  bfloat16 pos_qnan1 = absl::bit_cast<bfloat16>(static_cast<uint16_t>(0x7FE0u));
  bfloat16 pos_qnan2 = absl::bit_cast<bfloat16>(static_cast<uint16_t>(0x7FE1u));

  bfloat16 inf = absl::bit_cast<bfloat16>(static_cast<uint16_t>(0x7F80u));
  bfloat16 neg_inf = absl::bit_cast<bfloat16>(static_cast<uint16_t>(0xFF80u));
  bfloat16 pos_zero = absl::bit_cast<bfloat16>(static_cast<uint16_t>(0x0000u));
  bfloat16 neg_zero = absl::bit_cast<bfloat16>(static_cast<uint16_t>(0x8000u));

  // 9.183550e-41
  bfloat16 pos_denorm01 =
      absl::bit_cast<bfloat16>(static_cast<uint16_t>(0x0001u));
  // 1.836710e-40
  bfloat16 pos_denorm02 =
      absl::bit_cast<bfloat16>(static_cast<uint16_t>(0x0002u));
  // -9.183550e-41
  bfloat16 neg_denorm01 =
      absl::bit_cast<bfloat16>(static_cast<uint16_t>(0x8001u));
  // -1.836710e-40
  bfloat16 neg_denorm02 =
      absl::bit_cast<bfloat16>(static_cast<uint16_t>(0x8002u));

  bfloat16 v_neg_90 = static_cast<bfloat16>(-90.445f);
  bfloat16 v_pos_90 = static_cast<bfloat16>(90.445f);
  bfloat16 v_0_0125 = static_cast<bfloat16>(0.0125f);

  std::vector<bfloat16> input_row_start = {
      neg_inf,   inf,          neg_denorm01, pos_zero,     pos_qnan0,
      pos_qnan2, neg_zero,     v_neg_90,     inf,          neg_qnan0,
      neg_qnan2, v_neg_90,     neg_zero,     pos_denorm01, neg_denorm02,
      neg_qnan1, pos_zero,     neg_inf,      v_pos_90,     pos_qnan1,
      v_0_0125,  pos_denorm02, neg_denorm01, pos_denorm01, neg_denorm02};
  Array2D<bfloat16> input(8, 1024, neg_qnan0);
  for (int i = 0; i < input_row_start.size(); ++i) {
    for (int j = 0; j < 8; ++j) {
      input(j, i) = input_row_start[i];
    }
  }

  auto literal = LiteralUtil::CreateR2FromArray2D<bfloat16>(input);

  for (int k : {2, 4, 8, 12, 16, 24, 32, 64}) {
    const auto hlo_text_module = absl::StrFormat(R"(
      HloModule topk

      ENTRY TopK {
        x = bf16[8,1024]{1,0} parameter(0)
        ROOT topk = (bf16[8,%d]{1,0}, s32[8,%d]{1,0}) topk(x), k=%d, largest=true
      }
    )",
                                                 k, k, k);

    ASSERT_OK_AND_ASSIGN(auto module,
                         ParseAndReturnVerifiedModule(hlo_text_module));
    // Interpreter matches TotalOrder logic, ensuring exact equivalence.
    EXPECT_TRUE(RunAndCompare(std::move(module),
                              absl::Span<const Literal* const>{&literal},
                              ErrorSpec{0, 0}));
  }
}
}  // namespace
}  // namespace xla
