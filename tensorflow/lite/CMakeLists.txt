include_directories(tools/make/downloads/eigen)
include_directories(tools/make/downloads/gemmlowp)
include_directories(tools/make/downloads/farmhash/src)
include_directories(tools/make/downloads/flatbuffers/include)
include_directories(tools/make/downloads/absl)

add_definitions(-DEIGEN_MPL2_ONLY)
add_definitions(-fPIC)
add_definitions(-funsafe-math-optimizations)
add_definitions(-ftree-vectorize)
add_definitions(-flax-vector-conversions)
add_definitions(-fomit-frame-pointer)
add_definitions(-D__LE_NN__)
add_definitions(-D__ANDROID_API__=29)

set (TFLITE_TARGET_NAME tensorflow-lite)

FILE(GLOB TFLITE_SOURCE_FILES
    *.cc
    *.c
    c/*.c
    core/*.cc
    core/api/*.cc
    kernels/*.cc
    kernels/internal/*.cc
    kernels/internal/optimized/*.cc
    kernels/internal/reference/*.cc
    kernels/*.c
    kernels/internal/*.c
    kernels/internal/optimized/*.c
    kernels/internal/reference/*.c
    nnapi/*.cc
    tools/make/downloads/farmhash/src/farmhash.cc
    tools/make/downloads/fft2d/fftsg.c
    tools/make/downloads/fft2d/fftsg2d.c
    tools/make/downloads/flatbuffers/src/util.cpp
    tools/evaluation/utils.cc
    nnapi/nnapi_implementation.cc
    delegates/nnapi/nnapi_delegate.cc
    delegates/nnapi/quant_lstm_sup.cc
    profiling/time.cc
    experimental/c/*.c
    experimental/c/*.cc
    experimental/resource_variable/*.cc
    experimental/ruy/allocator.cc
    experimental/ruy/block_map.cc
    experimental/ruy/blocking_counter.cc
    experimental/ruy/context.cc
    experimental/ruy/detect_dotprod.cc
    experimental/ruy/kernel_arm32.cc
    experimental/ruy/kernel_arm64.cc
    experimental/ruy/pack_arm.cc
    experimental/ruy/pmu.cc
    experimental/ruy/thread_pool.cc
    experimental/ruy/trace.cc
    experimental/ruy/trmul.cc
    experimental/ruy/tune.cc
    experimental/ruy/wait.cc
)

list(FILTER TFLITE_SOURCE_FILES EXCLUDE REGEX ".*test.cc$")
list(FILTER TFLITE_SOURCE_FILES EXCLUDE REGEX ".*/*/*test.cc$")
list(FILTER TFLITE_SOURCE_FILES EXCLUDE REGEX ".*/*/*/*test.cc$")
list(FILTER TFLITE_SOURCE_FILES EXCLUDE REGEX ".*/*/*/*/*test.cc$")
list(FILTER TFLITE_SOURCE_FILES EXCLUDE REGEX ".*experimental/c/*test.cc$")
list(FILTER TFLITE_SOURCE_FILES EXCLUDE REGEX ".*/*test_util.cc$")
list(FILTER TFLITE_SOURCE_FILES EXCLUDE REGEX ".*/*test_.*.cc$")
list(FILTER TFLITE_SOURCE_FILES EXCLUDE REGEX ".*kernels/test_util.cc$")
list(FILTER TFLITE_SOURCE_FILES EXCLUDE REGEX ".*mmap_allocation_disabled.cc$")
list(FILTER TFLITE_SOURCE_FILES EXCLUDE REGEX ".*nnapi_delegate_disabled.cc$")
list(FILTER TFLITE_SOURCE_FILES EXCLUDE REGEX ".*minimal_logging_android.cc$")
list(FILTER TFLITE_SOURCE_FILES EXCLUDE REGEX ".*minimal_logging_ios.cc$")
list(FILTER TFLITE_SOURCE_FILES EXCLUDE REGEX ".*kernels/subgraph_test_util.cc$")
list(FILTER TFLITE_SOURCE_FILES EXCLUDE REGEX ".*nnapi/nnapi_implementation_disabled.cc$")

add_library(${TFLITE_TARGET_NAME} SHARED ${TFLITE_SOURCE_FILES})

target_link_libraries(${TFLITE_TARGET_NAME} pthread)
target_link_libraries(${TFLITE_TARGET_NAME} rt)
target_link_libraries(${TFLITE_TARGET_NAME} dl)
target_link_libraries(${TFLITE_TARGET_NAME} c)
target_link_libraries(${TFLITE_TARGET_NAME} m)
target_link_libraries(${TFLITE_TARGET_NAME} z)

install(
    TARGETS ${TFLITE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

install(
    DIRECTORY .
    DESTINATION include/tflite/tensorflow/lite
    FILES_MATCHING
    PATTERN "*.h"
)

########################### label_image #############################
set(LABEL_IMAGE_TARGET_NAME label_image)

set(LABEL_IMAGE_SOURCE_FILES
    examples/label_image/bitmap_helpers.cc
    examples/label_image/label_image.cc
)

add_executable(${LABEL_IMAGE_TARGET_NAME} ${LABEL_IMAGE_SOURCE_FILES})

target_link_libraries(${LABEL_IMAGE_TARGET_NAME} ${TFLITE_TARGET_NAME})

install(
    TARGETS ${LABEL_IMAGE_TARGET_NAME}
    RUNTIME DESTINATION ${TFLITE_INSTALL_BINDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### Minimal ################################
set(MINIMAL_TARGET_NAME minimal)

set(MINIMAL_SOURCE_FILES
    examples/minimal/minimal.cc
)

add_executable(${MINIMAL_TARGET_NAME} ${MINIMAL_SOURCE_FILES})

target_link_libraries(${MINIMAL_TARGET_NAME} ${TFLITE_TARGET_NAME})

install(
    TARGETS ${MINIMAL_TARGET_NAME}
    RUNTIME DESTINATION ${TFLITE_INSTALL_BINDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)
