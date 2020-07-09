########################### benchmark_utils #############################
set(BENCHMARK_UTILS_TARGET_NAME benchmark_utils)

set(BENCHMARK_UTILS_SOURCE_FILES
    benchmark_utils.cc
)

add_library(${BENCHMARK_UTILS_TARGET_NAME} SHARED ${BENCHMARK_UTILS_SOURCE_FILES})

target_link_libraries(${BENCHMARK_UTILS_TARGET_NAME}
    time
    memory_info
)

install(
    TARGETS ${BENCHMARK_UTILS_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### benchmark_params #############################
set(BENCHMARK_PARAMS_TARGET_NAME benchmark_params)

set(BENCHMARK_PARAMS_SOURCE_FILES
    benchmark_params.cc
)

add_library(${BENCHMARK_PARAMS_TARGET_NAME} SHARED ${BENCHMARK_PARAMS_SOURCE_FILES})

install(
    TARGETS ${BENCHMARK_PARAMS_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### benchmark_model_lib #############################
set(BENCHMARK_MODEL_LIB_TARGET_NAME benchmark_model_lib)

set(BENCHMARK_MODEL_LIB_SOURCE_FILES
    benchmark_model.cc
)

add_library(${BENCHMARK_MODEL_LIB_TARGET_NAME} SHARED ${BENCHMARK_MODEL_LIB_SOURCE_FILES})

target_link_libraries(${BENCHMARK_MODEL_LIB_TARGET_NAME}
    command_line_flags
    benchmark_utils
    benchmark_params
)

install(
    TARGETS ${BENCHMARK_MODEL_LIB_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### benchmark_tflite_model_lib #############################
set(BENCHMARK_TFLITE_MODEL_TARGET_NAME benchmark_tflite_model_lib)

set(BENCHMARK_TFLITE_MODEL_SOURCE_FILES
    benchmark_tflite_model.cc
    profiling_listener.cc
    nnapi_delegate_provider.cc
    gpu_delegate_provider.cc
    hexagon_delegate_provider.cc
    xnnpack_delegate_provider.cc
)

add_library(${BENCHMARK_TFLITE_MODEL_TARGET_NAME} SHARED ${BENCHMARK_TFLITE_MODEL_SOURCE_FILES})

target_link_libraries(${BENCHMARK_TFLITE_MODEL_TARGET_NAME} ${TFLITE_TARGET_NAME}
    absl::base
    absl::strings
    ruy_profiler
    profile_summarizer
    benchmark_model_lib
    xnnpack_delegate
    nnapi_delegate
    gpu_delegate
    gl_delegate
    evaluation_utils
    profile_summarizer_formatter
)

install(
    TARGETS ${BENCHMARK_TFLITE_MODEL_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### benchmark_performance_options #############################
set(BENCHMARK_PERFORMANCE_OPTIONS_TARGET_NAME benchmark_performance_options)

set(BENCHMARK_PERFORMANCE_OPTIONS_SOURCE_FILES
     benchmark_performance_options.cc
)

add_library(${BENCHMARK_PERFORMANCE_OPTIONS_TARGET_NAME} SHARED ${BENCHMARK_PERFORMANCE_OPTIONS_SOURCE_FILES})

target_link_libraries(${BENCHMARK_PERFORMANCE_OPTIONS_TARGET_NAME}
    benchmark_model_lib
    benchmark_utils
)

install(
    TARGETS ${BENCHMARK_PERFORMANCE_OPTIONS_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### benchmark_model #############################
set(BENCHMARK_MODEL_TARGET_NAME benchmark_model)

set(BENCHMARK_MODEL_TARGET_SOURCE_FILES
    benchmark_main.cc
)

add_executable(${BENCHMARK_MODEL_TARGET_NAME} ${BENCHMARK_MODEL_TARGET_SOURCE_FILES})

target_link_libraries(${BENCHMARK_MODEL_TARGET_NAME}
    benchmark_tflite_model_lib
)

install(
    TARGETS ${BENCHMARK_MODEL_TARGET_NAME}
    RUNTIME DESTINATION ${TFLITE_INSTALL_BINDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### benchmark_model_performance_options #############################
set(BENCHMARK_MODEL_PERFORMANCE_OPTIONS_TARGET_NAME benchmark_model_performance_options)

set(BENCHMARK_MODEL_PERFORMANCE_OPTIONS_SOURCE_FILES
    benchmark_tflite_performance_options_main.cc
)

add_executable(${BENCHMARK_MODEL_PERFORMANCE_OPTIONS_TARGET_NAME} ${BENCHMARK_MODEL_PERFORMANCE_OPTIONS_SOURCE_FILES})

target_link_libraries(${BENCHMARK_MODEL_PERFORMANCE_OPTIONS_TARGET_NAME}
    benchmark_performance_options
    benchmark_tflite_model_lib
)

install(
    TARGETS ${BENCHMARK_MODEL_PERFORMANCE_OPTIONS_TARGET_NAME}
    RUNTIME DESTINATION ${TFLITE_INSTALL_BINDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### benchmark_test ################################
set(BENCHMARK_TARGET_NAME benchmark_test)

set(BENCHMARK_SOURCE_FILES
    benchmark_test.cc
    benchmark_performance_options.cc
)

add_executable(${BENCHMARK_TARGET_NAME} ${BENCHMARK_SOURCE_FILES})

target_link_libraries(${BENCHMARK_TARGET_NAME}
    benchmark_tflite_model_lib
    evaluation_utils
    gtest
    absl::algorithm
)

install(
    TARGETS ${BENCHMARK_TARGET_NAME}
    RUNTIME DESTINATION ${TFLITE_INSTALL_BINDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### benchmark_utils_test ################################
set(BENCHMARK_UTILS_TARGET_NAME benchmark_utils_test)

set(BENCHMARK_UTILS_SOURCE_FILES
    benchmark_utils_test.cc
)

add_executable(${BENCHMARK_UTILS_TARGET_NAME} ${BENCHMARK_UTILS_SOURCE_FILES})

target_link_libraries(${BENCHMARK_UTILS_TARGET_NAME}
    time
    gtest
    gtest_main
    benchmark_utils
)

install(
    TARGETS ${BENCHMARK_UTILS_TARGET_NAME}
    RUNTIME DESTINATION ${TFLITE_INSTALL_BINDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)
