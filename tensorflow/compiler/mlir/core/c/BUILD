load("//tensorflow:tensorflow.bzl", "tf_copts")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "builtin_op_data.h",
        "tflite_types.h",
    ],
)

cc_library(
    name = "tflite_types",
    hdrs = [
        "tflite_types.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tf_copts(),
)

# LINT.IfChange(common)
cc_library(
    name = "tflite_common",
    srcs = [],
    hdrs = [
        "builtin_op_data.h",
        "tflite_types.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tf_copts(),
    visibility = [
        "//tensorflow/compiler/mlir/kernels:__pkg__",
        "//tensorflow/compiler/mlir/quantization/tensorflow/utils:__pkg__",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)
# LINT.ThenChange(//tensorflow/lite/core/c:common)
