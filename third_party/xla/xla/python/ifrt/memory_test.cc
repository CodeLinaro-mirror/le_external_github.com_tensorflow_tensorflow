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

#include "xla/python/ifrt/memory.h"

#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

using ::testing::Optional;

namespace xla {
namespace ifrt {
namespace {

TEST(MemoryKindTest, EqualityForUnspecified) {
  MemoryKind memory_kind1;
  MemoryKind memory_kind2;
  EXPECT_EQ(memory_kind1, memory_kind2);
}

TEST(MemoryKindTest, EqualityForSameString) {
  MemoryKind memory_kind1("abc");
  MemoryKind memory_kind2("abc");
  EXPECT_EQ(memory_kind1, memory_kind2);
}

TEST(MemoryKindTest, EqualityForSameStringContent) {
  MemoryKind memory_kind1("abc");
  MemoryKind memory_kind2(absl::StrCat("ab", "c"));
  EXPECT_EQ(memory_kind1, memory_kind2);
}

TEST(MemoryKindTest, InequalityForDifferentStringContent) {
  MemoryKind memory_kind1("abc");
  MemoryKind memory_kind2("def");
  EXPECT_NE(memory_kind1, memory_kind2);
}

TEST(MemoryKindTest, InequalityBetweenSpecifiedAndUnspecified) {
  {
    MemoryKind memory_kind1("abc");
    MemoryKind memory_kind2;
    EXPECT_NE(memory_kind1, memory_kind2);
  }
  {
    MemoryKind memory_kind1;
    MemoryKind memory_kind2("abc");
    EXPECT_NE(memory_kind1, memory_kind2);
  }
}

TEST(MemoryKindTest, MemorySafety) {
  auto memory_kind_str = std::make_unique<std::string>("abc");
  MemoryKind memory_kind(*memory_kind_str);

  memory_kind_str.reset();
  EXPECT_THAT(memory_kind.memory_kind(), Optional(absl::string_view("abc")));
}

TEST(MemoryKindTest, EqualityForUnspecifiedAndNullopt) {
  MemoryKind memory_kind1;
  MemoryKind memory_kind2(std::nullopt);
  EXPECT_EQ(memory_kind1, memory_kind2);
}

TEST(MemoryKindTest, DefaultMemoryKindIsDevice) {
  MemoryKind default_memory_kind;
  MemoryKind device_memory_kind("device");
  MemoryKind nullopt_memory_kind(std::nullopt);

  EXPECT_TRUE(default_memory_kind.is_default());
  EXPECT_TRUE(device_memory_kind.is_default());
  EXPECT_TRUE(nullopt_memory_kind.is_default());

  EXPECT_EQ(default_memory_kind, device_memory_kind);
  EXPECT_EQ(nullopt_memory_kind, device_memory_kind);

  EXPECT_EQ(default_memory_kind.memory_kind(), "device");
  EXPECT_EQ(absl::StrCat(default_memory_kind), "device");
  EXPECT_EQ(absl::Hash<MemoryKind>()(default_memory_kind),
            absl::Hash<MemoryKind>()(device_memory_kind));
}

TEST(MemoryKindTest, EmptyStringIsNotDefault) {
  MemoryKind empty_memory_kind("");
  MemoryKind default_memory_kind;
  EXPECT_FALSE(empty_memory_kind.is_default());
  EXPECT_NE(empty_memory_kind, default_memory_kind);
}

TEST(MemoryKindTest, MemoryKindViewCompatibility) {
  MemoryKind memory_kind("abc");
  auto view = memory_kind.memory_kind();

  // `std::optional<absl::string_view>`-like behavior.
  EXPECT_TRUE(view.has_value());
  EXPECT_TRUE(static_cast<bool>(view));
  EXPECT_EQ(*view, "abc");
  EXPECT_EQ(view->data(), view.data());
  EXPECT_EQ(view.value(), "abc");
  EXPECT_EQ(view.value_or("default"), "abc");
  std::optional<absl::string_view> opt_sv = view;
  EXPECT_EQ(opt_sv, "abc");

  // `absl::string_view`-like behavior.
  absl::string_view sv = view;
  EXPECT_EQ(sv, "abc");
  EXPECT_EQ(view.string_view(), "abc");
  EXPECT_EQ(view.size(), 3);
  EXPECT_EQ(view.length(), 3);
  EXPECT_FALSE(view.empty());
  EXPECT_EQ(view[0], 'a');
  EXPECT_EQ(view.front(), 'a');
  EXPECT_EQ(view.back(), 'c');
  EXPECT_EQ(view.substr(1, 2), "bc");

  // Comparisons.
  EXPECT_EQ(view, "abc");
  EXPECT_EQ("abc", view);
  EXPECT_NE(view, "def");
  EXPECT_NE("def", view);
  EXPECT_NE(view, std::nullopt);
  EXPECT_NE(std::nullopt, view);

  // Stringification and hashing.
  EXPECT_EQ(absl::StrCat(view), "abc");
  EXPECT_EQ(absl::Hash<MemoryKind::MemoryKindView>()(view),
            absl::Hash<absl::string_view>()("abc"));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
