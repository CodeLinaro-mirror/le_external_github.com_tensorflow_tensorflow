# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "host_allocator_extension",
    srcs = ["host_allocator_extension.cc"],
    hdrs = ["host_allocator_extension.h"],
    deps = ["//xla/pjrt/c:pjrt_c_api_hdrs"],
)

cc_library(
    name = "host_allocator_interface_impl",
    srcs = ["host_allocator_interface_impl.cc"],
    hdrs = ["host_allocator_interface_impl.h"],
    deps = [
        ":host_allocator_extension",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "@com_google_absl//absl/log:check",
    ],
)

xla_cc_test(
    name = "host_allocator_interface_impl_test",
    srcs = ["host_allocator_interface_impl_test.cc"],
    deps = [
        ":host_allocator_extension",
        ":host_allocator_interface_impl",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "@com_google_googletest//:gtest_main",
    ],
)
