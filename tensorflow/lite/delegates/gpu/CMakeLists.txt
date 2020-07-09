########################### gl_delegate #############################
set(GL_DELEGATE_TARGET_NAME gl_delegate)

set(GL_DELEGATE_SOURCE_FILES
    gl_delegate.cc
)

add_library(${GL_DELEGATE_TARGET_NAME} SHARED ${GL_DELEGATE_SOURCE_FILES})

target_compile_definitions(${GL_DELEGATE_TARGET_NAME} PRIVATE
  EGL_EGLEXT_PROTOTYPES
)

target_include_directories(${GL_DELEGATE_TARGET_NAME} PRIVATE
   ${CMAKE_BINARY_DIR}
   gl
   common
)

target_link_libraries(${GL_DELEGATE_TARGET_NAME} ${TFLITE_TARGET_NAME}
    absl::core_headers
    absl::span
    minimal_logging
    common_convert
    common_model_builder
    common_model_transformer
    common_shape
    general_transformations
    gl_api
    gl_command_queue
    gl_compiler
    egl_environment
    gl_request_gpu_info
    gl_object_manager
    bhwc_to_phwc4
    phwc4_to_bhwc
    gl_registry
    workgroups_best_effort_calculator
    flatbuffers
)

install(
    TARGETS ${GL_DELEGATE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)


########################### gpu_api #############################
set(GPU_API_TARGET_NAME gpu_api)

set(GPU_API_SOURCE_FILES
    api.cc
)

add_library(${GPU_API_TARGET_NAME} SHARED ${GPU_API_SOURCE_FILES})

target_include_directories(${GPU_API_TARGET_NAME} PUBLIC
    ${TFLITE_VULKAN_INC_DIR}
    gl
    common
)

target_link_libraries(${GPU_API_TARGET_NAME} ${TFLITE_TARGET_NAME}
    common_data_type
    OpenCL
    absl::span
    absl::variant
)

install(
    TARGETS ${GPU_API_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gpu_delegate #############################
set(GPU_DELEGATE_TARGET_NAME gpu_delegate)

set(GPU_DELEGATE_SOURCE_FILES
    delegate.cc
)

add_library(${GPU_DELEGATE_TARGET_NAME} SHARED ${GPU_DELEGATE_SOURCE_FILES})

target_include_directories(${GPU_DELEGATE_TARGET_NAME} PRIVATE
    cl
    gl
    common
)

target_link_libraries(${GPU_DELEGATE_TARGET_NAME} ${TFLITE_TARGET_NAME}
    gpu_api
    gl_api2
    cl_api
    opencl_wrapper
    cl_tensor_type_util
    common_shape
    common_model_builder
    common_model_transformer
    absl::memory
    absl::span
    absl::any
)

install(
    TARGETS ${GPU_DELEGATE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

add_subdirectory(cl)
add_subdirectory(common)
add_subdirectory(gl)
