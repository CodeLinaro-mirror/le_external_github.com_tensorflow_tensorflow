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

#include "xla/service/gpu/launch_dimensions.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {
using ::tsl::testing::IsOkAndHolds;

TEST(LaunchDimensionsTest, DefaultConstruction) {
  LaunchDimensions dimensions{};
  EXPECT_EQ(dimensions.block_counts().x, 1);
  EXPECT_EQ(dimensions.block_counts().y, 1);
  EXPECT_EQ(dimensions.block_counts().z, 1);
  EXPECT_EQ(dimensions.thread_counts_per_block().x, 1);
  EXPECT_EQ(dimensions.thread_counts_per_block().y, 1);
  EXPECT_EQ(dimensions.thread_counts_per_block().z, 1);
}

TEST(LaunchDimensionsTest, LinearConstruction) {
  constexpr int kNumBlocks = 42;
  constexpr int kNumThreads = 123;
  LaunchDimensions dimensions{kNumBlocks, kNumThreads};
  EXPECT_EQ(dimensions.block_counts().x, kNumBlocks);
  EXPECT_EQ(dimensions.block_counts().y, 1);
  EXPECT_EQ(dimensions.block_counts().z, 1);
  EXPECT_EQ(dimensions.thread_counts_per_block().x, kNumThreads);
  EXPECT_EQ(dimensions.thread_counts_per_block().y, 1);
  EXPECT_EQ(dimensions.thread_counts_per_block().z, 1);
}

TEST(LaunchDimensionsTest, ToProto) {
  se::BlockDim block_dimensions{1, 2, 3};
  se::ThreadDim thread_dimensions{4, 5, 6};
  LaunchDimensions dimensions{block_dimensions, thread_dimensions};

  LaunchDimensionsProto proto = dimensions.ToProto();
  EXPECT_TRUE(proto.has_block_counts());
  EXPECT_TRUE(proto.has_thread_counts_per_block());
  EXPECT_THAT(proto.block_counts(),
              tsl::proto_testing::EqualsProto(block_dimensions.ToProto()));
  EXPECT_THAT(proto.thread_counts_per_block(),
              tsl::proto_testing::EqualsProto(thread_dimensions.ToProto()));
}

TEST(LaunchDimensionsTest, ToAndFromProto) {
  se::BlockDim block_dimensions{10, 20, 30};
  se::ThreadDim thread_dimensions{44, 55, 66};
  LaunchDimensions dimensions{block_dimensions, thread_dimensions};

  EXPECT_THAT(LaunchDimensions::FromProto(dimensions.ToProto()),
              IsOkAndHolds(dimensions));
}

}  // namespace
}  // namespace xla::gpu
