"""
Provides the list of patches applied for files that are not exported inside google.

These are usually long-term duration patches that could not be applied in the previous copybara
workflow.
"""

excluded_files_patch_list = [
    "//third_party/triton/excluded:env_vars.patch",
    "//third_party/triton/excluded:sparse_dot_nvgpu.patch",
    "//third_party/triton/excluded:sparse_dot_base.patch",
    "//third_party/triton/excluded:sparse_dot_passes.patch",
]
