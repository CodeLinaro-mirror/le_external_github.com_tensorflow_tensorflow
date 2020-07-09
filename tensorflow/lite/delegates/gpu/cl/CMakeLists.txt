########################### cl_api #############################
set(CL_API_TARGET_NAME cl_api)

set(CL_API_SOURCE_FILES
    api.cc
)

add_library(${CL_API_TARGET_NAME} SHARED ${CL_API_SOURCE_FILES})

target_link_libraries(${CL_API_TARGET_NAME}
    cl_command_queue
    cl_event
    egl_sync
    cl_environment
    gl_interop
    inference_context
    opencl_wrapper
    cl_precision
    cl_tensor
    cl_tensor_type
    cl_tensor_type_util
    cl_converter
    common_data_type
    common_shape
    gpu_api
    absl::memory
    absl::variant
)

install(
    TARGETS ${CL_API_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_buffer #############################
set(CL_BUFFER_TARGET_NAME cl_buffer)

set(CL_BUFFER_SOURCE_FILES
    buffer.cc
)

add_library(${CL_BUFFER_TARGET_NAME} SHARED ${CL_BUFFER_SOURCE_FILES})

target_link_libraries(${CL_BUFFER_TARGET_NAME}
    cl_command_queue
    cl_context
    opencl_wrapper
    cl_util
    absl::strings
    absl::span
)

install(
    TARGETS ${CL_BUFFER_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_command_queue #############################
set(CL_COMMAND_QUEUE_TARGET_NAME cl_command_queue)

set(CL_COMMAND_QUEUE_SOURCE_FILES
    cl_command_queue.cc
)

add_library(${CL_COMMAND_QUEUE_TARGET_NAME} SHARED ${CL_COMMAND_QUEUE_SOURCE_FILES})

target_link_libraries(${CL_COMMAND_QUEUE_TARGET_NAME}
    cl_context
    cl_device
    cl_event
    cl_kernel
    opencl_wrapper
    cl_util
    absl::strings
    absl::time
)

install(
    TARGETS ${CL_COMMAND_QUEUE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_context #############################
set(CL_CONTEXT_TARGET_NAME cl_context)

set(CL_CONTEXT_SOURCE_FILES
    cl_context.cc
)

add_library(${CL_CONTEXT_TARGET_NAME} SHARED ${CL_CONTEXT_SOURCE_FILES})

target_link_libraries(${CL_CONTEXT_TARGET_NAME}
    cl_device
    cl_image_format
    opencl_wrapper
    cl_util
    common_data_type
    absl::strings
)

install(
    TARGETS ${CL_CONTEXT_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_device #############################
set(CL_DEVICE_TARGET_NAME cl_device)

set(CL_DEVICE_SOURCE_FILES
    cl_device.cc
)

add_library(${CL_DEVICE_TARGET_NAME} SHARED ${CL_DEVICE_SOURCE_FILES})

target_link_libraries(${CL_DEVICE_TARGET_NAME}
    opencl_wrapper
    cl_util
    absl::strings
)

install(
    TARGETS ${CL_DEVICE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_event #############################
set(CL_EVENT_TARGET_NAME cl_event)

set(CL_EVENT_SOURCE_FILES
    cl_event.cc
)

add_library(${CL_EVENT_TARGET_NAME} SHARED ${CL_EVENT_SOURCE_FILES})

target_link_libraries(${CL_EVENT_TARGET_NAME}
    opencl_wrapper
)

install(
    TARGETS ${CL_EVENT_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_image_format #############################
set(CL_IMAGE_FORMAT_TARGET_NAME cl_image_format)

set(CL_IMAGE_FORMAT_SOURCE_FILES
    cl_image_format.cc
)

add_library(${CL_IMAGE_FORMAT_TARGET_NAME} SHARED ${CL_IMAGE_FORMAT_SOURCE_FILES})

target_link_libraries(${CL_IMAGE_FORMAT_TARGET_NAME}
    opencl_wrapper
    common_data_type
)

install(
    TARGETS ${CL_IMAGE_FORMAT_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_kernel #############################
set(CL_KERNEL_TARGET_NAME cl_kernel)

set(CL_KERNEL_SOURCE_FILES
    cl_kernel.cc
)

add_library(${CL_KERNEL_TARGET_NAME} SHARED ${CL_KERNEL_SOURCE_FILES})

target_link_libraries(${CL_KERNEL_TARGET_NAME}
    cl_context
    cl_device
    cl_program
    opencl_wrapper
    cl_util
    cl_flt_type
    absl::strings
)

install(
    TARGETS ${CL_KERNEL_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_memory #############################
set(CL_MEMORY_TARGET_NAME cl_memory)

set(CL_MEMORY_SOURCE_FILES
    cl_memory.cc
)

add_library(${CL_MEMORY_TARGET_NAME} SHARED ${CL_MEMORY_SOURCE_FILES})

target_link_libraries(${CL_MEMORY_TARGET_NAME}
    opencl_wrapper
)

install(
    TARGETS ${CL_MEMORY_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_program #############################
set(CL_PROGRAM_TARGET_NAME cl_program)

set(CL_PROGRAM_SOURCE_FILES
    cl_program.cc
)

add_library(${CL_PROGRAM_TARGET_NAME} SHARED ${CL_PROGRAM_SOURCE_FILES})

target_link_libraries(${CL_PROGRAM_TARGET_NAME}
    cl_context
    cl_device
    opencl_wrapper
    cl_util
    absl::strings
    absl::span
)

install(
    TARGETS ${CL_PROGRAM_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### egl_sync #############################
set(EGL_SYNC_TARGET_NAME egl_sync)

set(EGL_SYNC_SOURCE_FILES
    egl_sync.cc
)

add_library(${EGL_SYNC_TARGET_NAME} SHARED ${EGL_SYNC_SOURCE_FILES})

target_compile_definitions(${EGL_SYNC_TARGET_NAME} PRIVATE
  EGL_EGLEXT_PROTOTYPES
)

target_link_libraries(${EGL_SYNC_TARGET_NAME}
    cl_device
    gl_errors
)

install(
    TARGETS ${EGL_SYNC_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_environment #############################
set(CL_ENVIRONMENT_TARGET_NAME cl_environment)

set(CL_ENVIRONMENT_SOURCE_FILES
    environment.cc
)

add_library(${CL_ENVIRONMENT_TARGET_NAME} SHARED ${CL_ENVIRONMENT_SOURCE_FILES})

target_link_libraries(${CL_ENVIRONMENT_TARGET_NAME}
    cl_command_queue
    cl_context
    cl_device
    cl_kernel
    cl_precision
    cl_program_cache
    cl_tensor
    cl_tensor_type
    cl_util
    common_data_type
    common_shape
)

install(
    TARGETS ${CL_ENVIRONMENT_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_interop #############################
set(GL_INTEROP_TARGET_NAME gl_interop)

set(GL_INTEROP_SOURCE_FILES
    gl_interop.cc
)

add_library(${GL_INTEROP_TARGET_NAME} SHARED ${GL_INTEROP_SOURCE_FILES})

target_link_libraries(${GL_INTEROP_TARGET_NAME}
    cl_context
    cl_device
    cl_event
    cl_memory
    egl_sync
    cl_environment
    opencl_wrapper
    gl_sync
    absl::strings
)

install(
    TARGETS ${GL_INTEROP_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gpu_api_delegate #############################
set(GPU_API_DELEGATE_TARGET_NAME gpu_api_delegate)

set(GPU_API_DELEGATE_SOURCE_FILES
    gpu_api_delegate.cc
)

add_library(${GPU_API_DELEGATE_TARGET_NAME} SHARED ${GPU_API_DELEGATE_SOURCE_FILES})

target_link_libraries(${GPU_API_DELEGATE_TARGET_NAME} ${TFLITE_TARGET_NAME}
    cl_api
    opencl_wrapper
    cl_tensor_type_util
    gpu_api
    gpu_delegate
    common_model_builder
    common_model_transformer
    general_transformations
    absl::span
)

install(
    TARGETS ${GPU_API_DELEGATE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### inference_context #############################
set(INFERENCE_CONTEXT_TARGET_NAME inference_context)

set(INFERENCE_CONTEXT_SOURCE_FILES
    inference_context.cc
)

add_library(${INFERENCE_CONTEXT_TARGET_NAME} SHARED ${INFERENCE_CONTEXT_SOURCE_FILES})

target_link_libraries(${INFERENCE_CONTEXT_TARGET_NAME}
    cl_buffer
    cl_command_queue
    cl_device
    cl_environment
    opencl_wrapper
    cl_precision
    cl_storage_type_util
    cl_tensor_type
    cl_gpu_operation
    operation_selector
    common_data_type
    common_memory_management
    common_model_transformer
    common_operations
    common_shape
    transformations_add_bias
    transformations_merge_padding_with
)

install(
    TARGETS ${INFERENCE_CONTEXT_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_linear_storage #############################
set(LINEAR_STORAGE_TARGET_NAME cl_linear_storage)

set(LINEAR_STORAGE_SOURCE_FILES
    linear_storage.cc
)

add_library(${LINEAR_STORAGE_TARGET_NAME} SHARED ${LINEAR_STORAGE_SOURCE_FILES})

target_link_libraries(${LINEAR_STORAGE_TARGET_NAME}
    cl_buffer
    opencl_wrapper
    cl_tensor_type
    cl_texture2d
    cl_util
    common_data_type
    absl::strings
    absl::span
)

install(
    TARGETS ${LINEAR_STORAGE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### opencl_wrapper #############################
set(OPENCL_WRAPPER_TARGET_NAME opencl_wrapper)

set(OPENCL_WRAPPER_SOURCE_FILES
    opencl_wrapper.cc
)

add_library(${OPENCL_WRAPPER_TARGET_NAME} SHARED ${OPENCL_WRAPPER_SOURCE_FILES})

target_link_libraries(${OPENCL_WRAPPER_TARGET_NAME} ${TFLITE_TARGET_NAME}
    absl::strings
    OpenCL
)

install(
    TARGETS ${OPENCL_WRAPPER_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_precision #############################
set(PRECISION_TARGET_NAME cl_precision)

set(PRECISION_SOURCE_FILES
    precision.cc
)

add_library(${PRECISION_TARGET_NAME} SHARED ${PRECISION_SOURCE_FILES})

target_link_libraries(${PRECISION_TARGET_NAME}
    common_data_type
)

install(
    TARGETS ${PRECISION_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### compiled_program_cache_cc_fbs ###########################
set(COMPILED_PROGRAM_CACHE_CC_FBS_TARGET compiled_program_cache_cc_fbs)

set(COMPILED_PROGRAM_CACHE_CC_FBS_SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/compiled_program_cache.fbs
)

set(COMPILED_PROGRAM_CACHE_CC_FBS_OUTPUT
   compiled_program_cache_generated.h
)

add_custom_target(
  ${COMPILED_PROGRAM_CACHE_CC_FBS_TARGET} ALL
  DEPENDS ${COMPILED_PROGRAM_CACHE_CC_FBS_OUTPUT}
)

add_custom_command(
  OUTPUT ${COMPILED_PROGRAM_CACHE_CC_FBS_OUTPUT}
  COMMAND ${SYSROOT_BINDIR_NATIVE}/flatc
      -c --scoped-enums
      ${COMPILED_PROGRAM_CACHE_CC_FBS_SOURCE}
  DEPENDS ${COMPILED_PROGRAM_CACHE_CC_FBS_SOURCE}
  COMMENT ""
)

########################### cl_program_cache #############################
set(PROGRAM_CACHE_TARGET_NAME cl_program_cache)

set(PROGRAM_CACHE_SOURCE_FILES
    program_cache.cc
)

add_library(${PROGRAM_CACHE_TARGET_NAME} SHARED ${PROGRAM_CACHE_SOURCE_FILES})

add_dependencies(${PROGRAM_CACHE_TARGET_NAME} ${COMPILED_PROGRAM_CACHE_CC_FBS_TARGET})

target_include_directories(${PROGRAM_CACHE_TARGET_NAME} PRIVATE ${CMAKE_BINARY_DIR})

target_link_libraries(${PROGRAM_CACHE_TARGET_NAME}
    cl_context
    cl_device
    cl_kernel
    cl_program
    cl_util
    absl::span
    flatbuffers
)

install(
    TARGETS ${PROGRAM_CACHE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_storage_type_util #############################
set(STORAGE_TYPE_UTIL_TARGET_NAME cl_storage_type_util)

set(STORAGE_TYPE_UTIL_SOURCE_FILES
    storage_type_util.cc
)

add_library(${STORAGE_TYPE_UTIL_TARGET_NAME} SHARED ${STORAGE_TYPE_UTIL_SOURCE_FILES})

target_link_libraries(${STORAGE_TYPE_UTIL_TARGET_NAME} ${TFLITE_TARGET_NAME}
    cl_context
    cl_device
    cl_tensor_type
    common_data_type
    common_shape
)

install(
    TARGETS ${STORAGE_TYPE_UTIL_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_tensor #############################
set(TENSOR_TARGET_NAME cl_tensor)

set(TENSOR_SOURCE_FILES
    tensor.cc
)

add_library(${TENSOR_TARGET_NAME} SHARED ${TENSOR_SOURCE_FILES})

target_link_libraries(${TENSOR_TARGET_NAME}
    cl_command_queue
    cl_context
    cl_device
    cl_image_format
    cl_memory
    cl_tensor_type
    cl_util
    common_data_type
    common_shape
    absl::strings
    absl::span
)

install(
    TARGETS ${TENSOR_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_tensor_type #############################
set(TENSOR_TYPE_TARGET_NAME cl_tensor_type)

set(TENSOR_TYPE_SOURCE_FILES
    tensor_type.cc
)

add_library(${TENSOR_TYPE_TARGET_NAME} SHARED ${TENSOR_TYPE_SOURCE_FILES})

target_link_libraries(${TENSOR_TYPE_TARGET_NAME}
    common_data_type
    common_shape
)

install(
    TARGETS ${TENSOR_TYPE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_tensor_type_util #############################
set(TENSOR_TYPE_UTIL_TARGET_NAME cl_tensor_type_util)

set(TENSOR_TYPE_UTIL_SOURCE_FILES
    tensor_type_util.cc
)

add_library(${TENSOR_TYPE_UTIL_TARGET_NAME} SHARED ${TENSOR_TYPE_UTIL_SOURCE_FILES})

target_link_libraries(${TENSOR_TYPE_UTIL_TARGET_NAME}
    cl_tensor_type
    gpu_api
)

install(
    TARGETS ${TENSOR_TYPE_UTIL_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_texture2d #############################
set(TEXTURE2D_TARGET_NAME cl_texture2d)

set(TEXTURE2D_SOURCE_FILES
    texture2d.cc
)

add_library(${TEXTURE2D_TARGET_NAME} SHARED ${TEXTURE2D_SOURCE_FILES})

target_link_libraries(${TEXTURE2D_TARGET_NAME}
    cl_command_queue
    cl_context
    opencl_wrapper
    cl_tensor_type
    cl_util
    common_data_type
    absl::strings
    absl::span
)

install(
    TARGETS ${TEXTURE2D_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_util #############################
set(CL_UTIL_TARGET_NAME cl_util)

set(CL_UTIL_SOURCE_FILES
    util.cc
)

add_library(${CL_UTIL_TARGET_NAME} SHARED ${CL_UTIL_SOURCE_FILES})

target_link_libraries(${CL_UTIL_TARGET_NAME}
    opencl_wrapper
    common_data_type
    absl::strings
    absl::span
)

install(
    TARGETS ${CL_UTIL_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

add_subdirectory(kernels)
add_subdirectory(selectors)