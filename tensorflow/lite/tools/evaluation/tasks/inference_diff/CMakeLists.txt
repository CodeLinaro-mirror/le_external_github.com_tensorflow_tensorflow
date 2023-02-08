# The Inference Diff tool for Tensorflow Lite.

populate_source_vars("${TFLITE_SOURCE_DIR}/tools/evaluation/tasks/inference_diff"
  TFLITE_INFERENCE_DIFF_SRCS
)
list(APPEND TFLITE_INFERENCE_DIFF_SRCS
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
  list(APPEND TFLITE_INFERENCE_DIFF_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/xnnpack_delegate_provider.cc
  )
else()
  set(TFLITE_INFERENCE_DIFF_CC_OPTIONS "-DTFLITE_WITHOUT_XNNPACK")
endif()  # TFLITE_ENABLE_XNNPACK

if(CMAKE_SYSTEM_NAME MATCHES "Android" OR CMAKE_SYSTEM_NAME MATCHES "Linux")
  if(_TFLITE_ENABLE_NNAPI)
    list(APPEND TFLITE_INFERENCE_DIFF_SRCS
      ${TFLITE_SOURCE_DIR}/tools/delegates/nnapi_delegate_provider.cc
    )
  endif()  # _TFLITE_ENABLE_NNAPI
endif()  # Android

if(TFLITE_ENABLE_GPU)
  list(APPEND TFLITE_INFERENCE_DIFF_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/gpu_delegate_provider.cc
  )
endif()  # TFLITE_ENABLE_GPU

if(TFLITE_ENABLE_HEXAGON)
  list(APPEND TFLITE_INFERENCE_DIFF_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/hexagon_delegate_provider.cc
  )
endif()  # TFLITE_ENABLE_HEXAGON

add_executable(inf_diff_run_eval
  EXCLUDE_FROM_ALL
  ${TFLITE_INFERENCE_DIFF_SRCS}
)

install(TARGETS inf_diff_run_eval
  RUNTIME
)

target_compile_options(inf_diff_run_eval
  PRIVATE
    ${TFLITE_INFERENCE_DIFF_CC_OPTIONS}
)

target_link_libraries(inf_diff_run_eval
  PUBLIC
    evaluation_proto
    inference_profiler_stage
    tflite_inference_stage
    ${CMAKE_DL_LIBS}
)

if (TFLITE_C_BUILD_SHARED_LIBS)
    target_link_libraries(inf_diff_run_eval
      PUBLIC
        tensorflowlite_c
    )
else ()
    target_link_libraries(inf_diff_run_eval
      PUBLIC
        tensorflow-lite
    )
endif()
