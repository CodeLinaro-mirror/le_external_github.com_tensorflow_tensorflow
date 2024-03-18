licenses(["restricted"])  # NVIDIA proprietary license

filegroup(
    name = "include",
    srcs = glob([
        "include/**",
    ]),
)

filegroup(
    name = "nvvm",
    srcs = [
        "nvvm/libdevice/libdevice.10.bc",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "bin",
    srcs = glob([
        "bin/**",
        "nvvm/bin/**",
    ]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    hdrs = [":include"],
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
