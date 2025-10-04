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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/codegen/intrinsic/cpp/eigen_unary_ll.h"

namespace xla::codegen {
namespace {
using ::testing::ContainsRegex;
using ::testing::Not;

TEST(EigenUnaryTest, FastTanhfIsVectorized) {
  const std::string arm = llvm_ir::kEigenUnaryLlArmIr;
  EXPECT_THAT(arm, ContainsRegex("fmul <4 x float>"));
  EXPECT_THAT(arm, ContainsRegex("<4 x float>.*0x3E4DF2A3C0000000"));
  EXPECT_THAT(arm, Not(ContainsRegex("llvm.x86")));
  EXPECT_THAT(arm, ContainsRegex("llvm.aarch64.neon"));

  const std::string x86 = llvm_ir::kEigenUnaryLlX86Ir;
  EXPECT_THAT(x86, ContainsRegex("fmul <4 x float>"));
  EXPECT_THAT(x86, ContainsRegex("<4 x float>.*0x3E4DF2A3C0000000"));
  EXPECT_THAT(x86, ContainsRegex("llvm.x86"));
  EXPECT_THAT(x86, Not(ContainsRegex("llvm.aarch64")));
}

}  // namespace
}  // namespace xla::codegen
