########################### common_cc_fbs ###########################
set(COMMON_CC_FBS_TARGET common_cc_fbs)

set(COMMON_CC_FBS_SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/common.fbs
)

set(COMMON_CC_FBS_OUTPUT
   common_generated.h
)

add_custom_target(
  ${COMMON_CC_FBS_TARGET} ALL
  DEPENDS ${COMMON_CC_FBS_OUTPUT}
)

add_custom_command(
  OUTPUT ${COMMON_CC_FBS_OUTPUT}
  COMMAND ${SYSROOT_BINDIR_NATIVE}/flatc
      -c
      ${COMMON_CC_FBS_SOURCE}
  DEPENDS ${COMMON_CC_FBS_SOURCE}
  COMMENT ""
)

########################### compiled_model_cc_fbs ###########################
set(COMPILED_MODEL_CC_FBS_TARGET compiled_model_cc_fbs)

set(COMPILED_MODEL_CC_FBS_SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/compiled_model.fbs
)

set(COMPILED_MODEL_CC_FBS_OUTPUT
   compiled_model_generated.h
)

add_custom_target(
  ${COMPILED_MODEL_CC_FBS_TARGET} ALL
  DEPENDS ${COMPILED_MODEL_CC_FBS_OUTPUT}
)

add_custom_command(
  OUTPUT ${COMPILED_MODEL_CC_FBS_OUTPUT}
  COMMAND ${SYSROOT_BINDIR_NATIVE}/flatc
      -c --scoped-enums
      ${COMPILED_MODEL_CC_FBS_SOURCE}
  DEPENDS ${COMPILED_MODEL_CC_FBS_SOURCE}
  COMMENT ""
)

########################### metadata_cc_fbs ###########################
set(METADATA_CC_FBS_TARGET metadata_cc_fbs)

set(METADATA_CC_FBS_SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/metadata.fbs
)

set(METADATA_CC_FBS_OUTPUT
   metadata_generated.h
)

add_custom_target(
  ${METADATA_CC_FBS_TARGET} ALL
  DEPENDS ${METADATA_CC_FBS_OUTPUT}
)

add_custom_command(
  OUTPUT ${METADATA_CC_FBS_OUTPUT}
  COMMAND ${SYSROOT_BINDIR_NATIVE}/flatc
      -c
      ${METADATA_CC_FBS_SOURCE}
  DEPENDS ${METADATA_CC_FBS_SOURCE}
  COMMENT ""
)

########################### workgroups_cc_fbs ###########################
set(WORKGROUPS_CC_FBS_TARGET workgroups_cc_fbs)

set(WORKGROUPS_CC_FBS_SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/workgroups.fbs
)

set(WORKGROUPS_CC_FBS_OUTPUT
   workgroups_generated.h
)

add_custom_target(
  ${WORKGROUPS_CC_FBS_TARGET} ALL
  DEPENDS ${WORKGROUPS_CC_FBS_OUTPUT}
)

add_custom_command(
  OUTPUT ${WORKGROUPS_CC_FBS_OUTPUT}
  COMMAND ${SYSROOT_BINDIR_NATIVE}/flatc
      -c
      ${WORKGROUPS_CC_FBS_SOURCE}
  DEPENDS ${WORKGROUPS_CC_FBS_SOURCE}
  COMMENT ""
)

########################### gl_api #############################
set(GL_API_TARGET_NAME gl_api)

set(GL_API_SOURCE_FILES
    api.cc
)

add_library(${GL_API_TARGET_NAME} SHARED ${GL_API_SOURCE_FILES})

add_dependencies(${GL_API_TARGET_NAME} ${COMMON_CC_FBS_TARGET})

target_include_directories(${GL_API_TARGET_NAME} PRIVATE ${CMAKE_BINARY_DIR})

target_link_libraries(${GL_API_TARGET_NAME}
    gl_command_queue
    gl_compiler
    gl_request_gpu_info
    gl_object_manager
    gl_runtime
    gl_serialization
    absl::memory
    absl::strings
    common_data_type
    common_shape
    workgroups_calculator
)

install(
    TARGETS ${GL_API_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_api2 #############################
set(GL_API2_TARGET_NAME gl_api2)

set(GL_API2_SOURCE_FILES
    api2.cc
)

add_library(${GL_API2_TARGET_NAME} SHARED ${GL_API2_SOURCE_FILES})

target_link_libraries(${GL_API2_TARGET_NAME}
    gl_command_queue
    gl_compiler
    egl_environment
    gl_request_gpu_info
    gl_runtime
    gpu_api
    common_data_type
    common_shape
    gl_converter
    gl_registry
    workgroups_default_calculator
    absl::memory
    absl::span
)

install(
    TARGETS ${GL_API2_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_command_queue #############################
set(GL_COMMAND_QUEUE_TARGET_NAME gl_command_queue)

set(GL_COMMAND_QUEUE_SOURCE_FILES
    command_queue.cc
)

add_library(${GL_COMMAND_QUEUE_TARGET_NAME} SHARED ${GL_COMMAND_QUEUE_SOURCE_FILES})

target_link_libraries(${GL_COMMAND_QUEUE_TARGET_NAME}
    gl_program
    gl_sync
    common_gpu_info
    absl::memory
)

install(
    TARGETS ${GL_COMMAND_QUEUE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_compiler #############################
set(GL_COMPILER_TARGET_NAME gl_compiler)

set(GL_COMPILER_SOURCE_FILES
    compiler.cc
)

add_library(${GL_COMPILER_TARGET_NAME} SHARED ${GL_COMPILER_SOURCE_FILES})

target_link_libraries(${GL_COMPILER_TARGET_NAME}
    common_data_type
    common_gpu_info
    common_shape
    common_model_transformer
    common_operations
    gl_float16_conversions
    gl_compiled_node
    gl_fuse_auto_input
    gl_fuse_inline
    gl_fuse_inplace
    gl_shader_codegen
    absl::memory
    absl::any
)

install(
    TARGETS ${GL_COMPILER_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### egl_context #############################
set(EGL_CONTEXT_TARGET_NAME egl_context)

set(EGL_CONTEXT_SOURCE_FILES
    egl_context.cc
)

add_library(${EGL_CONTEXT_TARGET_NAME} SHARED ${EGL_CONTEXT_SOURCE_FILES})

target_link_libraries(${EGL_CONTEXT_TARGET_NAME}
     gl_errors
)

install(
    TARGETS ${EGL_CONTEXT_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### egl_environment #############################
set(EGL_ENVIRONMENT_TARGET_NAME egl_environment)

set(EGL_ENVIRONMENT_SOURCE_FILES
    egl_environment.cc
)

add_library(${EGL_ENVIRONMENT_TARGET_NAME} SHARED ${EGL_ENVIRONMENT_SOURCE_FILES})

target_link_libraries(${EGL_ENVIRONMENT_TARGET_NAME}
    egl_context
    egl_surface
    gl_request_gpu_info
    absl::memory
)

install(
    TARGETS ${EGL_ENVIRONMENT_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### egl_surface #############################
set(EGL_SURFACE_TARGET_NAME egl_surface)

set(EGL_SURFACE_SOURCE_FILES
    egl_surface.cc
)

add_library(${EGL_SURFACE_TARGET_NAME} SHARED ${EGL_SURFACE_SOURCE_FILES})

target_link_libraries(${EGL_SURFACE_TARGET_NAME}
    gl_errors
)

install(
    TARGETS ${EGL_SURFACE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_float16_conversions #############################
set(GL_FLOAT16_CONV_TARGET_NAME gl_float16_conversions)

set(GL_FLOAT16_CONV_SOURCE_FILES
    float16_conversions.cc
)

add_library(${GL_FLOAT16_CONV_TARGET_NAME} SHARED ${GL_FLOAT16_CONV_SOURCE_FILES})

target_link_libraries(${GL_FLOAT16_CONV_TARGET_NAME}
    common_data_type
    common_shape
    absl::variant
)

install(
    TARGETS ${GL_FLOAT16_CONV_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_buffer #############################
set(GL_BUFFER_TARGET_NAME gl_buffer)

set(GL_BUFFER_SOURCE_FILES
    gl_buffer.cc
)

add_library(${GL_BUFFER_TARGET_NAME} SHARED ${GL_BUFFER_SOURCE_FILES})

target_link_libraries(${GL_BUFFER_TARGET_NAME}
    gl_errors
    absl::span
)

install(
    TARGETS ${GL_BUFFER_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_errors #############################
set(GL_ERROR_TARGET_NAME gl_errors)

set(GL_ERROR_SOURCE_FILES
    gl_errors.cc
)

add_library(${GL_ERROR_TARGET_NAME} SHARED ${GL_ERROR_SOURCE_FILES})

target_link_libraries(${GL_ERROR_TARGET_NAME}
    EGL_adreno
    GLESv2_adreno
    absl::strings
)

install(
    TARGETS ${GL_ERROR_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_program #############################
set(GL_PROGRAM_TARGET_NAME gl_program)

set(GL_PROGRAM_SOURCE_FILES
    gl_program.cc
)

add_library(${GL_PROGRAM_TARGET_NAME} SHARED ${GL_PROGRAM_SOURCE_FILES})

target_link_libraries(${GL_PROGRAM_TARGET_NAME}
     gl_errors
     gl_shader
    absl::variant
)

install(
    TARGETS ${GL_PROGRAM_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_shader #############################
set(GL_SHADER_TARGET_NAME gl_shader)

set(GL_SHADER_SOURCE_FILES
    gl_shader.cc
)

add_library(${GL_SHADER_TARGET_NAME} SHARED ${GL_SHADER_SOURCE_FILES})

target_link_libraries(${GL_SHADER_TARGET_NAME}
    gl_errors
)

install(
    TARGETS ${GL_SHADER_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_texture #############################
set(GL_TEXTURE_TARGET_NAME gl_texture)

set(GL_TEXTURE_SOURCE_FILES
    gl_texture.cc
)

add_library(${GL_TEXTURE_TARGET_NAME} SHARED ${GL_TEXTURE_SOURCE_FILES})

target_link_libraries(${GL_TEXTURE_TARGET_NAME}
    gl_errors
    common_data_type
    common_shape
    absl::span
)

install(
    TARGETS ${GL_TEXTURE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_sync #############################
set(GL_SYNC_TARGET_NAME gl_sync)

set(GL_SYNC_SOURCE_FILES
    gl_sync.cc
)

add_library(${GL_SYNC_TARGET_NAME} SHARED ${GL_SYNC_SOURCE_FILES})

target_link_libraries(${GL_SYNC_TARGET_NAME}
     gl_buffer
     gl_errors
     gl_program
)

install(
    TARGETS ${GL_SYNC_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_object_manager #############################
set(GL_OBJ_MANAGER_TARGET_NAME gl_object_manager)

set(GL_OBJ_MANAGER_SOURCE_FILES
    object_manager.cc
)

add_library(${GL_OBJ_MANAGER_TARGET_NAME} SHARED ${GL_OBJ_MANAGER_SOURCE_FILES})

target_link_libraries(${GL_OBJ_MANAGER_TARGET_NAME}
    gl_buffer
    gl_texture
    common_convert
    common_data_type
    common_shape
    absl::memory
    absl::span
)

install(
    TARGETS ${GL_OBJ_MANAGER_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_request_gpu_info #############################
set(GL_REQUEST_GPU_INFO_TARGET_NAME gl_request_gpu_info)

set(GL_REQUEST_GPU_INFO_SOURCE_FILES
    request_gpu_info.cc
)

add_library(${GL_REQUEST_GPU_INFO_TARGET_NAME} SHARED ${GL_REQUEST_GPU_INFO_SOURCE_FILES})

target_link_libraries(${GL_REQUEST_GPU_INFO_TARGET_NAME}
    gl_errors
    common_gpu_info
    absl::strings
)

install(
    TARGETS ${GL_REQUEST_GPU_INFO_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_runtime #############################
set(GL_RUNTIME_TARGET_NAME gl_runtime)

set(GL_RUNTIME_SOURCE_FILES
    runtime.cc
)

add_library(${GL_RUNTIME_TARGET_NAME} SHARED ${GL_RUNTIME_SOURCE_FILES})

target_link_libraries(${GL_RUNTIME_TARGET_NAME}
    gl_command_queue
    gl_buffer
    gl_errors
    gl_program
    gl_shader
    gl_texture
    gl_object_manager
    common_data_type
    common_gpu_info
    common_memory_management
    absl::strings
)

install(
    TARGETS ${GL_RUNTIME_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_serialization #############################
set(GL_SERIALIZATION_TARGET_NAME gl_serialization)

set(GL_SERIALIZATION_SOURCE_FILES
    serialization.cc
)

add_library(${GL_SERIALIZATION_TARGET_NAME} SHARED ${GL_SERIALIZATION_SOURCE_FILES})

add_dependencies(${GL_SERIALIZATION_TARGET_NAME}
    ${COMMON_CC_FBS_TARGET}
    ${COMPILED_MODEL_CC_FBS_TARGET}
)

target_include_directories(${GL_SERIALIZATION_TARGET_NAME} PRIVATE ${CMAKE_BINARY_DIR})

target_link_libraries(${GL_SERIALIZATION_TARGET_NAME}
    common_data_type
    absl::span
    absl::variant
    flatbuffers
)

install(
    TARGETS ${GL_SERIALIZATION_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)
add_subdirectory(compiler)
add_subdirectory(converters)
add_subdirectory(kernels)
add_subdirectory(workgroups)

