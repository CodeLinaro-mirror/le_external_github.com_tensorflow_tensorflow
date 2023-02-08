# The Image Classification tool for Tensorflow Lite.

populate_source_vars("${TFLITE_SOURCE_DIR}/tools/evaluation/tasks/imagenet_image_classification"
  TFLITE_IMAGE_CLASSIFICATION_SRCS
)
list(APPEND TFLITE_IMAGE_CLASSIFICATION_SRCS
  ${TF_SOURCE_DIR}/tsl/util/stats_calculator.cc
  ${TFLITE_SOURCE_DIR}/profiling/memory_info.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summarizer.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summary_formatter.cc
  ${TFLITE_SOURCE_DIR}/profiling/time.cc
  ${TFLITE_SOURCE_DIR}/tools/command_line_flags.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/default_execution_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/delegate_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/evaluation/tasks/task_executor_main.cc
  ${TFLITE_SOURCE_DIR}/tools/evaluation/tasks/task_executor.cc
  ${TFLITE_SOURCE_DIR}/tools/evaluation/utils.cc
  ${TFLITE_SOURCE_DIR}/tools/evaluation/evaluation_delegate_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/tool_params.cc
  ${TFLITE_SOURCE_DIR}/delegates/utils/simple_delegate.cc
)

if(TFLITE_ENABLE_XNNPACK)
  list(APPEND TFLITE_IMAGE_CLASSIFICATION_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/xnnpack_delegate_provider.cc
  )
else()
  set(TFLITE_IMAGE_CLASSIFICATION_CC_OPTIONS "-DTFLITE_WITHOUT_XNNPACK")
endif()  # TFLITE_ENABLE_XNNPACK

if(CMAKE_SYSTEM_NAME MATCHES "Android" OR CMAKE_SYSTEM_NAME MATCHES "Linux")
  if(_TFLITE_ENABLE_NNAPI)
    list(APPEND TFLITE_IMAGE_CLASSIFICATION_SRCS
      ${TFLITE_SOURCE_DIR}/tools/delegates/nnapi_delegate_provider.cc
    )
  endif()  # _TFLITE_ENABLE_NNAPI
endif()  # Android

if(TFLITE_ENABLE_GPU)
  list(APPEND TFLITE_IMAGE_CLASSIFICATION_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/gpu_delegate_provider.cc
  )
endif()  # TFLITE_ENABLE_GPU

if(TFLITE_ENABLE_HEXAGON)
  list(APPEND TFLITE_IMAGE_CLASSIFICATION_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/hexagon_delegate_provider.cc
  )
endif()  # TFLITE_ENABLE_HEXAGON

add_executable(image_classify_run_eval
  EXCLUDE_FROM_ALL
  ${TFLITE_IMAGE_CLASSIFICATION_SRCS}
)

install(TARGETS image_classify_run_eval
  RUNTIME
)

target_compile_options(image_classify_run_eval
  PRIVATE
    ${TFLITE_IMAGE_CLASSIFICATION_CC_OPTIONS}
)

target_link_libraries(image_classify_run_eval
  PUBLIC
    evaluation_proto
    image_classification_stage
    ${CMAKE_DL_LIBS}
)

if (TFLITE_C_BUILD_SHARED_LIBS)
    target_link_libraries(image_classify_run_eval
      PUBLIC
        tensorflowlite_c
    )
else ()
    target_link_libraries(image_classify_run_eval
      PUBLIC
        tensorflow-lite
    )
endif()
