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

# TODO(unda): describe this package.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_copts")
load(
    "//tensorflow/cc/framework/fuzzing:op_fuzzing.bzl",
    "tf_gen_op_wrappers_fuzz",
)

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "cc_op_fuzz_gen_main",
    srcs = [
        "cc_op_fuzz_gen.cc",
        "cc_op_fuzz_gen.h",
        "cc_op_fuzz_gen_main.cc",
    ],
    copts = tf_copts(),
    data = [
        "//tensorflow/core/api_def:base_api_def",
    ],
    deps = [
        "//tensorflow/cc:cc_op_gen_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:hash",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/platform:status",
    ],
)

# copybara:uncomment_begin
# tf_gen_op_wrappers_fuzz(
#     name = "array_ops_fuzz",
#     api_def_srcs = ["//tensorflow/core/api_def:base_api_def"],
#     kernel_deps = [
#         "//tensorflow/c/kernels:bitcast_op",
#         "//tensorflow/core/kernels:array",
#         "//tensorflow/core/kernels:check_numerics_op",
#         "//tensorflow/core/kernels:fake_quant_ops",
#         "//tensorflow/core/kernels:quantized_ops",
#         "//tensorflow/core/kernels:scatter_nd_op",
#         "//tensorflow/core/kernels/image:extract_image_patches_op",
#         "//tensorflow/core/kernels/image:extract_volume_patches_op",
#         "//tensorflow/core/kernels/image:mirror_pad_op",
#         "//tensorflow/core/kernels/linalg:matrix_band_part_op",
#         "//tensorflow/core/kernels/linalg:matrix_diag_op",
#         "//tensorflow/core/kernels/linalg:matrix_set_diag_op",
#     ],
#     op_def_src = "//tensorflow/core/ops:array_ops_op_lib",
# )
# copybara:uncomment_end

bzl_library(
    name = "op_fuzzing_bzl",
    srcs = ["op_fuzzing.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow:tensorflow_bzl",
        "//tensorflow/core/platform:rules_cc_bzl",
    ],
)
