package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# LINT.IfChange(config_setting)
config_setting(
    name = "tflite_debug_delegate",
    define_values = {"tflite_debug_delegate": "true"},
)
# LINT.ThenChange(//tensorflow/lite/delegates/BUILD)
