########################### transformations_add_bias #############################
set(TRF_ADD_BIAS_TARGET_NAME transformations_add_bias)

set(TRF_ADD_BIAS_SOURCE_FILES
    add_bias.cc
)

add_library(${TRF_ADD_BIAS_TARGET_NAME} SHARED ${TRF_ADD_BIAS_SOURCE_FILES})

target_link_libraries(${TRF_ADD_BIAS_TARGET_NAME}
    common_data_type
    common_model_transformer
    common_operations
    absl::memory
    absl::strings
    absl::any
)

install(
    TARGETS ${TRF_ADD_BIAS_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### transformations_fuse_add_to_conv #############################
set(TRF_FUSE_ADD_TO_CONV_TARGET_NAME transformations_fuse_add_to_conv)

set(TRF_FUSE_ADD_TO_CONV_SOURCE_FILES
    fuse_add_to_conv.cc
)

add_library(${TRF_FUSE_ADD_TO_CONV_TARGET_NAME} SHARED ${TRF_FUSE_ADD_TO_CONV_SOURCE_FILES})

target_link_libraries(${TRF_FUSE_ADD_TO_CONV_TARGET_NAME}
    common_data_type
    common_model_transformer
    common_operations
    absl::strings
)

install(
    TARGETS ${TRF_FUSE_ADD_TO_CONV_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### transformations_fuse_mul_to_conv #############################
set(TRF_FUSE_MUL_TO_CONV_TARGET_NAME transformations_fuse_mul_to_conv)

set(TRF_FUSE_MUL_TO_CONV_SOURCE_FILES
    fuse_mul_to_conv.cc
)

add_library(${TRF_FUSE_MUL_TO_CONV_TARGET_NAME} SHARED ${TRF_FUSE_MUL_TO_CONV_SOURCE_FILES})

target_link_libraries(${TRF_FUSE_MUL_TO_CONV_TARGET_NAME}
    common_data_type
    common_model_transformer
    common_operations
)

install(
    TARGETS ${TRF_FUSE_MUL_TO_CONV_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### general_transformations #############################
set(TRF_GEN_TARGET_NAME general_transformations)

set(TRF_GEN_SOURCE_FILES
    general_transformations.cc
)

add_library(${TRF_GEN_TARGET_NAME} SHARED ${TRF_GEN_SOURCE_FILES})

target_link_libraries(${TRF_GEN_TARGET_NAME}
    transformations_fuse_add_to_conv
    transformations_fuse_mul_to_conv
    transformations_make_fully_connected
    transformations_make_padding
    transformations_match_dilated_convolution
    transformations_merge_padding_with
    transformations_remove_noop
    common_model_transformer
)

install(
    TARGETS ${TRF_GEN_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### transformations_make_fully_connected #############################
set(TRF_MAKE_FULLY_CON_TARGET_NAME transformations_make_fully_connected)

set(TRF_MAKE_FULLY_CON_SOURCE_FILES
    make_fully_connected.cc
)

add_library(${TRF_MAKE_FULLY_CON_TARGET_NAME} SHARED ${TRF_MAKE_FULLY_CON_SOURCE_FILES})

target_link_libraries(${TRF_MAKE_FULLY_CON_TARGET_NAME}
    common_model_transformer
    common_operations
    absl::memory
    absl::any
)

install(
    TARGETS ${TRF_MAKE_FULLY_CON_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### transformations_make_padding #############################
set(TRF_MAKE_PAD_TARGET_NAME transformations_make_padding)

set(TRF_MAKE_PAD_SOURCE_FILES
    make_padding.cc
)

add_library(${TRF_MAKE_PAD_TARGET_NAME} SHARED ${TRF_MAKE_PAD_SOURCE_FILES})

target_link_libraries(${TRF_MAKE_PAD_TARGET_NAME}
    common_model_transformer
    common_operations
    absl::memory
    absl::any
)

install(
    TARGETS ${TRF_MAKE_PAD_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### transformations_match_dilated_convolution #############################
set(TRF_DILATED_CONV_TARGET_NAME transformations_match_dilated_convolution)

set(TRF_DILATED_CONV_SOURCE_FILES
    match_dilated_convolution.cc
)

add_library(${TRF_DILATED_CONV_TARGET_NAME} SHARED ${TRF_DILATED_CONV_SOURCE_FILES})

target_link_libraries(${TRF_DILATED_CONV_TARGET_NAME}
    common_model_transformer
    common_operations
    absl::memory
    absl::any
)

install(
    TARGETS ${TRF_DILATED_CONV_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### transformations_merge_padding_with #############################
set(TRF_MERGE_PAD_WITH_TARGET_NAME transformations_merge_padding_with)

set(TRF_MERGE_PAD_WITH_SOURCE_FILES
    merge_padding_with.cc
)

add_library(${TRF_MERGE_PAD_WITH_TARGET_NAME} SHARED ${TRF_MERGE_PAD_WITH_SOURCE_FILES})

target_link_libraries(${TRF_MERGE_PAD_WITH_TARGET_NAME}
    common_data_type
    common_model_transformer
    common_operations
    absl::memory
    absl::strings
    absl::any
)

install(
    TARGETS ${TRF_MERGE_PAD_WITH_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### transformations_remove_noop #############################
set(TRF_REMOVE_NOOP_TARGET_NAME transformations_remove_noop)

set(TRF_REMOVE_NOOP_SOURCE_FILES
    remove_noop.cc
)

add_library(${TRF_REMOVE_NOOP_TARGET_NAME} SHARED ${TRF_REMOVE_NOOP_SOURCE_FILES})

target_link_libraries(${TRF_REMOVE_NOOP_TARGET_NAME}
    common_data_type
    common_model_transformer
    common_operations
    absl::memory
)

install(
    TARGETS ${TRF_REMOVE_NOOP_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)