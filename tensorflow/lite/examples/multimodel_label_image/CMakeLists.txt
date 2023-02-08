#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The multimodel_label_image example for Tensorflow Lite.

populate_source_vars("${TFLITE_SOURCE_DIR}/examples/multimodel_label_image"
  TFLITE_MULTIMODEL_LABEL_IMAGE_SRCS
  FILTER "_test\\.cc$"
)
list(APPEND TFLITE_MULTIMODEL_LABEL_IMAGE_SRCS
  ${TSL_SOURCE_DIR}/tsl/util/stats_calculator.cc
  ${TFLITE_SOURCE_DIR}/profiling/memory_info.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summarizer.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summary_formatter.cc
  ${TFLITE_SOURCE_DIR}/profiling/time.cc
  ${TFLITE_SOURCE_DIR}/tools/command_line_flags.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/default_execution_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/delegate_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/evaluation/utils.cc
  ${TFLITE_SOURCE_DIR}/tools/tool_params.cc
  ${TFLITE_SOURCE_DIR}/delegates/utils/simple_delegate.cc
  ${TFLITE_SOURCE_DIR}/examples/label_image/bitmap_helpers.cc
)


if(TFLITE_ENABLE_XNNPACK)
  list(APPEND TFLITE_MULTIMODEL_LABEL_IMAGE_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/xnnpack_delegate_provider.cc
  )
else()
  set(TFLITE_MULTIMODEL_LABEL_IMAGE_CC_OPTIONS "-DTFLITE_WITHOUT_XNNPACK")
endif()  # TFLITE_ENABLE_XNNPACK

if(CMAKE_SYSTEM_NAME MATCHES "Android" OR CMAKE_SYSTEM_NAME MATCHES "Linux")
  if(_TFLITE_ENABLE_NNAPI)
    list(APPEND TFLITE_MULTIMODEL_LABEL_IMAGE_SRCS
      ${TFLITE_SOURCE_DIR}/tools/delegates/nnapi_delegate_provider.cc
    )
  endif()  # _TFLITE_ENABLE_NNAPI
endif()  # Android

if(TFLITE_ENABLE_GPU)
  list(APPEND TFLITE_MULTIMODEL_LABEL_IMAGE_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/gpu_delegate_provider.cc
  )
endif()  # TFLITE_ENABLE_GPU

if(TFLITE_ENABLE_HEXAGON)
  list(APPEND TFLITE_MULTIMODEL_LABEL_IMAGE_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/hexagon_delegate_provider.cc
  )
endif()  # TFLITE_ENABLE_HEXAGON

add_executable(multimodel_label_image
  EXCLUDE_FROM_ALL
  ${TFLITE_MULTIMODEL_LABEL_IMAGE_SRCS}
)

install(TARGETS multimodel_label_image
  RUNTIME
)

target_compile_options(multimodel_label_image
  PRIVATE
    ${TFLITE_MULTIMODEL_LABEL_IMAGE_CC_OPTIONS}
)

if (TFLITE_C_BUILD_SHARED_LIBS)
  target_link_libraries(multimodel_label_image
    tensorflowlite_c
    ${CMAKE_DL_LIBS}
    )
else ()
  target_link_libraries(multimodel_label_image
    tensorflow-lite
    ${CMAKE_DL_LIBS}
    )
endif()
