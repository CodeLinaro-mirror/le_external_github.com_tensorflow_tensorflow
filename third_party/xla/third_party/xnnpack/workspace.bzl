"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "f16ff87fe1261cd3ec42cd6057c0e81096057508a7474523242c034fafdd5da2",
        strip_prefix = "XNNPACK-37e59cdb3b42c8cd9083e7cde65aef0f6e945b76",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/37e59cdb3b42c8cd9083e7cde65aef0f6e945b76.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
