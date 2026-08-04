/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/gpu_module_plugin.h"

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/configuration.pb.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/acceleration/configuration/proto_to_flatbuffer.h"

namespace tflite {
namespace acceleration {
namespace {

TEST(GpuModulePluginTest, DlopenFlags) {
  proto::ComputeSettings settings_proto;
  settings_proto.mutable_tflite_settings()->set_delegate(proto::GPU);
  settings_proto.mutable_tflite_settings()
      ->mutable_stable_delegate_loader_settings()
      ->set_delegate_path("invalid_path_to_force_dlopen_fail.so");

  flatbuffers::FlatBufferBuilder fbb;
  const ComputeSettings* settings = ConvertFromProto(settings_proto, &fbb);
  ASSERT_NE(settings, nullptr);
  ASSERT_NE(settings->tflite_settings(), nullptr);

  auto plugin = GpuModulePlugin::New(*settings->tflite_settings());
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
