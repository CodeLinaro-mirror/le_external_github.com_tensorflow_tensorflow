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

#include "xla/stream_executor/launch_dim.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace {
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

TEST(Dim3DTest, ToProto) {
  Dim3D dimensions{1, 2, 3};
  Dim3DProto proto = dimensions.ToProto();
  EXPECT_EQ(proto.x(), dimensions.x);
  EXPECT_EQ(proto.y(), dimensions.y);
  EXPECT_EQ(proto.z(), dimensions.z);
}

TEST(Dim3DTest, FromProto) {
  Dim3DProto proto;
  proto.set_x(1);
  proto.set_y(2);
  proto.set_z(3);
  TF_ASSERT_OK_AND_ASSIGN(Dim3D dimensions, Dim3D::FromProto(proto));
  EXPECT_EQ(proto.x(), dimensions.x);
  EXPECT_EQ(proto.y(), dimensions.y);
  EXPECT_EQ(proto.z(), dimensions.z);

  proto.set_x(-1);
  EXPECT_THAT(Dim3D::FromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ThreadDimTest, ToAndFromProto) {
  ThreadDim thread_dimensions{2, 4, 6};
  EXPECT_THAT(ThreadDim::FromProto(thread_dimensions.ToProto()),
              IsOkAndHolds(thread_dimensions));
}

TEST(BlockDimTest, ToAndFromProto) {
  BlockDim block_dimensions{2, 4, 6};
  EXPECT_THAT(BlockDim::FromProto(block_dimensions.ToProto()),
              IsOkAndHolds(block_dimensions));
}

TEST(ClusterDimTest, ToAndFromProto) {
  ClusterDim cluster_dimensions{2, 4, 6};
  EXPECT_THAT(ClusterDim::FromProto(cluster_dimensions.ToProto()),
              IsOkAndHolds(cluster_dimensions));
}

}  // namespace
}  // namespace stream_executor
