/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/*
 *Changes from Qualcomm Innovation Center, Inc. are provided under the following license:
 *
 *Copyright (c) 2021 Qualcomm Innovation Center, Inc.
 *
 *Redistribution and use in source and binary forms, with or without
 *modification, are permitted (subject to the limitations in the
 *disclaimer below) provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Qualcomm Innovation Center nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
 * GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
 * HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_UTILS_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_UTILS_H_

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#if !TFLITE_WITH_STABLE_ABI
// TODO(b/240438534): enable nnapi.
#if defined(__ANDROID__)
#define TFLITE_SUPPORTS_NNAPI_DELEGATE 1
#define TFLITE_SUPPORTS_GPU_DELEGATE 1
#elif (defined(CL_DELEGATE_NO_GL) || defined(LINUX))
#define TFLITE_SUPPORTS_GPU_DELEGATE 1
#endif  // defined(__ANDROID__)
#endif  // TFLITE_WITH_STABLE_ABI

// XNNPACK does not support s390x
// (see <https://github.com/tensorflow/tensorflow/pull/51655>).
#ifdef __s390x__
#define TFLITE_WITHOUT_XNNPACK 1
#endif

#if TFLITE_SUPPORTS_GPU_DELEGATE
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#if TFLITE_ENABLE_HEXAGON
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"
#endif

#if TFLITE_SUPPORTS_NNAPI_DELEGATE
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif  // TFLITE_SUPPORTS_NNAPI_DELEGATE

#ifndef TFLITE_WITHOUT_XNNPACK
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif  // !defined(TFLITE_WITHOUT_XNNPACK)

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace evaluation {

// Same as Interpreter::TfLiteDelegatePtr, defined here to avoid pulling
// in tensorflow/lite/interpreter.h dependency.
using TfLiteDelegatePtr =
    std::unique_ptr<TfLiteOpaqueDelegate, void (*)(TfLiteOpaqueDelegate*)>;

std::string StripTrailingSlashes(const std::string& path);

bool ReadFileLines(const std::string& file_path,
                   std::vector<std::string>* lines_output);

// If extension set is empty, all files will be listed. The strings in
// extension set are expected to be in lowercase and include the dot.
TfLiteStatus GetSortedFileNames(
    const std::string& directory, std::vector<std::string>* result,
    const std::unordered_set<std::string>& extensions);

inline TfLiteStatus GetSortedFileNames(const std::string& directory,
                                       std::vector<std::string>* result) {
  return GetSortedFileNames(directory, result,
                            std::unordered_set<std::string>());
}

// Returns nullptr on error, e.g. if NNAPI isn't supported on this platform.
TfLiteDelegatePtr CreateNNAPIDelegate();
#if TFLITE_SUPPORTS_NNAPI_DELEGATE
TfLiteDelegatePtr CreateNNAPIDelegate(StatefulNnApiDelegate::Options options);
#endif  // TFLITE_SUPPORTS_NNAPI_DELEGATE

TfLiteDelegatePtr CreateGPUDelegate();
#if TFLITE_SUPPORTS_GPU_DELEGATE
TfLiteDelegatePtr CreateGPUDelegate(TfLiteGpuDelegateOptionsV2* options);
#endif  // TFLITE_SUPPORTS_GPU_DELEGATE

TfLiteDelegatePtr CreateHexagonDelegate(
    const std::string& library_directory_path, bool profiling);
#if TFLITE_ENABLE_HEXAGON
TfLiteDelegatePtr CreateHexagonDelegate(
    const TfLiteHexagonDelegateOptions* options,
    const std::string& library_directory_path);
#endif

#ifndef TFLITE_WITHOUT_XNNPACK
TfLiteXNNPackDelegateOptions XNNPackDelegateOptionsDefault();
TfLiteDelegatePtr CreateXNNPACKDelegate();
TfLiteDelegatePtr CreateXNNPACKDelegate(
    const TfLiteXNNPackDelegateOptions* options);
#endif  // !defined(TFLITE_WITHOUT_XNNPACK)
TfLiteDelegatePtr CreateXNNPACKDelegate(int num_threads);

TfLiteDelegatePtr CreateCoreMlDelegate();
}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_UTILS_H_
