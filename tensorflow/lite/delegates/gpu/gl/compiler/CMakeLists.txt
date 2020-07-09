########################### gl_preprocessor #############################
set(GL_PREPROCESSOR_TARGET_NAME gl_preprocessor)

set(GL_PREPROCESSOR_SOURCE_FILES
    preprocessor.cc
)

add_library(${GL_PREPROCESSOR_TARGET_NAME} SHARED ${GL_PREPROCESSOR_SOURCE_FILES})

target_link_libraries(${GL_PREPROCESSOR_TARGET_NAME}
    absl::strings
)

install(
    TARGETS ${GL_PREPROCESSOR_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_object_accessor #############################
set(GL_OBJ_ACCESSOR_TARGET_NAME gl_object_accessor)

set(GL_OBJ_ACCESSOR_SOURCE_FILES
    object_accessor.cc
)

add_library(${GL_OBJ_ACCESSOR_TARGET_NAME} SHARED ${GL_OBJ_ACCESSOR_SOURCE_FILES})

target_link_libraries(${GL_OBJ_ACCESSOR_TARGET_NAME}
    gl_preprocessor
    gl_variable_accessor
    common_data_type
    common_shape
    absl::strings
    absl::str_format
    absl::variant
)

install(
    TARGETS ${GL_OBJ_ACCESSOR_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_shader_codegen #############################
set(GL_SHADER_CODEGEN_TARGET_NAME gl_shader_codegen)

set(GL_SHADER_CODEGEN_SOURCE_FILES
    shader_codegen.cc
)

add_library(${GL_SHADER_CODEGEN_TARGET_NAME} SHARED ${GL_SHADER_CODEGEN_SOURCE_FILES})

target_link_libraries(${GL_SHADER_CODEGEN_TARGET_NAME}
    gl_compiled_node
    gl_object_accessor
    gl_preprocessor
    gl_variable_accessor
    common_gpu_info
    common_data_type
    common_shape
    absl::strings
)

install(
    TARGETS ${GL_SHADER_CODEGEN_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_compiled_node #############################
set(GL_COMPILED_NODE_TARGET_NAME gl_compiled_node)

set(GL_COMPILED_NODE_SOURCE_FILES
    compiled_node.cc
)

add_library(${GL_COMPILED_NODE_TARGET_NAME} SHARED ${GL_COMPILED_NODE_SOURCE_FILES})

target_link_libraries(${GL_COMPILED_NODE_TARGET_NAME}
    gl_rename
    common_gpu_info
    absl::strings
)

install(
    TARGETS ${GL_COMPILED_NODE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_fuse_inplace #############################
set(GL_FUSE_INPLACE_TARGET_NAME gl_fuse_inplace)

set(GL_FUSE_INPLACE_SOURCE_FILES
    fuse_inplace.cc
)

add_library(${GL_FUSE_INPLACE_TARGET_NAME} SHARED ${GL_FUSE_INPLACE_SOURCE_FILES})

target_link_libraries(${GL_FUSE_INPLACE_TARGET_NAME}
    gl_compiled_node
    gl_preprocessor
    common_data_type
    common_shape
    common_model_transformer
    common_gpu_info
    absl::strings
    absl::any
)

install(
    TARGETS ${GL_FUSE_INPLACE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_fuse_inline #############################
set(GL_FUSE_INLINE_TARGET_NAME gl_fuse_inline)

set(GL_FUSE_INLINE_SOURCE_FILES
    fuse_inline.cc
)

add_library(${GL_FUSE_INLINE_TARGET_NAME} SHARED ${GL_FUSE_INLINE_SOURCE_FILES})

target_link_libraries(${GL_FUSE_INLINE_TARGET_NAME}
    gl_compiled_node
    common_data_type
    common_shape
    common_model_transformer
    common_gpu_info
    absl::strings
    absl::any
)

install(
    TARGETS ${GL_FUSE_INLINE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_rename #############################
set(GL_RENAME_TARGET_NAME gl_rename)

set(GL_RENAME_SOURCE_FILES
    rename.cc
)

add_library(${GL_RENAME_TARGET_NAME} SHARED ${GL_RENAME_SOURCE_FILES})

target_link_libraries(${GL_RENAME_TARGET_NAME}
    gl_object_accessor
    gl_preprocessor
    gl_variable_accessor
    common_data_type
    common_shape
    common_gpu_info
    absl::strings
)

install(
    TARGETS ${GL_RENAME_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_fuse_auto_input #############################
set(GL_FUSE_AUTO_INPUT_TARGET_NAME gl_fuse_auto_input)

set(GL_FUSE_AUTO_INPUT_SOURCE_FILES
    fuse_auto_input.cc
)

add_library(${GL_FUSE_AUTO_INPUT_TARGET_NAME} SHARED ${GL_FUSE_AUTO_INPUT_SOURCE_FILES})

target_link_libraries(${GL_FUSE_AUTO_INPUT_TARGET_NAME}
    gl_compiled_node
    common_data_type
    common_shape
    common_model_transformer
    common_gpu_info
    absl::strings
    absl::any
    absl::variant
)

install(
    TARGETS ${GL_FUSE_AUTO_INPUT_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_variable_accessor #############################
set(GL_VAR_ACCESSOR_TARGET_NAME gl_variable_accessor)

set(GL_VAR_ACCESSOR_SOURCE_FILES
    variable_accessor.cc
)

add_library(${GL_VAR_ACCESSOR_TARGET_NAME} SHARED ${GL_VAR_ACCESSOR_SOURCE_FILES})

target_link_libraries(${GL_VAR_ACCESSOR_TARGET_NAME}
    gl_preprocessor
    absl::strings
    absl::str_format
    absl::variant
)

install(
    TARGETS ${GL_VAR_ACCESSOR_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)