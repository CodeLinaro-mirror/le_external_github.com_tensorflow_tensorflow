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
# =============================================================================

"""Compute Library"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "compute_library",
        patch_file = [
            "//third_party/compute_library:acl_gemm_scheduling_heuristic.patch",
            "//third_party/compute_library:acl_stateless_gemm_workspace.patch",
            "//third_party/compute_library:compute_library.patch",
            "//third_party/compute_library:exclude_omp_scheduler.patch",
            "//third_party/compute_library:include_string.patch",
            "//third_party/compute_library:rules_python.patch",
        ],
        sha256 = "8273f68cd0bb17e9231a11a6618d245eb6d623884ae681c00e7a4eabca2dad42",
        strip_prefix = "ComputeLibrary-24.12",
        urls = tf_mirror_urls("https://github.com/ARM-software/ComputeLibrary/archive/refs/tags/v24.12.tar.gz"),
    )
