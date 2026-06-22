# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//xla/tsl/profiler/builds:build_config.bzl",
    "tf_profiler_copts",
    "tf_profiler_pybind_cc_library_wrapper",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla/tsl/profiler:internal"]),
    licenses = ["notice"],
)

# Linked to pywrap_tensorflow.
cc_library(
    name = "profiler_service_impl",
    srcs = ["profiler_service_impl.cc"],
    hdrs = ["profiler_service_impl.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//tensorflow/core/data/service:__pkg__",
        "//tensorflow/core/distributed_runtime/rpc:__pkg__",
        "//tensorflow/core/profiler/rpc:__pkg__",
        "//tensorflow/python:__pkg__",
        "//xla/tsl/profiler/rpc/client:__pkg__",
        "//tensorflow_serving/model_servers:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
        "//third_party/xprof/pywrap:__pkg__",
        "//xla/tsl/profiler:friends",
    ]),
    deps = [
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_time",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/profiler/rpc/client:save_profile",
        "//xla/tsl/profiler/utils:math_utils",
        "//xla/tsl/profiler/utils:time_utils",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/profiler/lib:profiler_session",
        "@tsl//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
        "@tsl//tsl/profiler/protobuf:profiler_service_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

tf_profiler_pybind_cc_library_wrapper(
    name = "profiler_server_for_pybind",
    actual = ":profiler_server_impl",
    visibility = internal_visibility(["//tensorflow/python/profiler/internal:__pkg__"]),
)

cc_library(
    name = "profiler_server_impl",
    srcs = ["profiler_server.cc"],
    hdrs = ["profiler_server.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla:__subpackages__",
        "//tensorflow/core/profiler/rpc:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler/rpc/client:__pkg__",
        "//third_party/xprof/pywrap:__pkg__",
        "//xla/tsl/profiler:friends",
    ]),
    deps = [
        ":profiler_service_impl",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:types",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
    ],
    alwayslink = True,
)
