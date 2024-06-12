"""Internal versions of TF Lite build rules."""

load("//tensorflow:tensorflow.bzl", "clean_dep")

# LINT.IfChange(tflite_copts_extra)
def tflite_copts_extra():
    """Defines extra Google-only compile time flags for tflite_copts()."""

    return select({
        clean_dep("//tensorflow:chromiumos_armv7"): [
            "-Wno-frame-larger-than",
        ],
        clean_dep("//tensorflow:linux_x86_64"): [
            "-fno-sanitize=shift-base",  # Possible invalid left shift in neon2sse.
        ],
        clean_dep("//tools/cc_target_os:mpu64"): [
            "-D__NO_MALLINFO__",
        ],
        "//conditions:default": [],
    })

# LINT.ThenChange(//tensorflow/lite:special_rules.bzl")
