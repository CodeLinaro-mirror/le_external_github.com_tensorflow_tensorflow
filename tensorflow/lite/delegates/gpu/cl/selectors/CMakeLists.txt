########################### convolution_selector #############################
set(CONV_SELECTOR_TARGET_NAME convolution_selector)

set(CONV_SELECTOR_SOURCE_FILES
    convolution_selector.cc
)

add_library(${CONV_SELECTOR_TARGET_NAME} SHARED ${CONV_SELECTOR_SOURCE_FILES})

target_link_libraries(${CONV_SELECTOR_TARGET_NAME}
    cl_tensor_type
    cl_conv_buffer_1x1
    cl_conv_constants
    cl_conv_powervr
    cl_conv_texture
    cl_gpu_operation
    cl_work_group_picking
    common_operations
    common_shape
    absl::memory
)

install(
    TARGETS ${CONV_SELECTOR_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### convolution_transposed_selector #############################
set(CONV_TRANSPOSED_SELECTOR_TARGET_NAME convolution_transposed_selector)

set(CONV_TRANSPOSED_SELECTOR_SOURCE_FILES
    convolution_transposed_selector.cc
)

add_library(${CONV_TRANSPOSED_SELECTOR_TARGET_NAME} SHARED ${CONV_TRANSPOSED_SELECTOR_SOURCE_FILES})

target_link_libraries(${CONV_TRANSPOSED_SELECTOR_TARGET_NAME}
    cl_tensor_type
    cl_convolution_transposed
    cl_convolution_transposed_3x3
    cl_convolution_transposed_3x3_thin
    cl_convolution_transposed_4x4
    cl_convolution_transposed_thin
    cl_gpu_operation
    common_operations
    absl::memory
)

install(
    TARGETS ${CONV_TRANSPOSED_SELECTOR_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### dw_convolution_selector #############################
set(DW_CONV_SELECTOR_TARGET_NAME dw_convolution_selector)

set(DW_CONV_SELECTOR_SOURCE_FILES
    dw_convolution_selector.cc
)

add_library(${DW_CONV_SELECTOR_TARGET_NAME} SHARED ${DW_CONV_SELECTOR_SOURCE_FILES})

target_link_libraries(${DW_CONV_SELECTOR_TARGET_NAME}
    cl_device
    cl_depth_wise_conv
    cl_depth_wise_conv_3x3
    cl_gpu_operation
    common_operations
    absl::memory
)

install(
    TARGETS ${DW_CONV_SELECTOR_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### fully_connected_selector #############################
set(FULLY_CONNECTED_SELECTOR_TARGET_NAME fully_connected_selector)

set(FULLY_CONNECTED_SELECTOR_SOURCE_FILES
    fully_connected_selector.cc
)

add_library(${FULLY_CONNECTED_SELECTOR_TARGET_NAME} SHARED ${FULLY_CONNECTED_SELECTOR_SOURCE_FILES})

target_link_libraries(${FULLY_CONNECTED_SELECTOR_TARGET_NAME}
    cl_conv_buffer_1x1
    cl_conv_powervr
    cl_conv_texture
    cl_fully_connected
    cl_gpu_operation
    common_operations
    absl::memory
)

install(
    TARGETS ${FULLY_CONNECTED_SELECTOR_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### simple_selectors #############################
set(SIMPLE_SELECTOR_TARGET_NAME simple_selectors)

set(SIMPLE_SELECTOR_SOURCE_FILES
    simple_selectors.cc
)

add_library(${SIMPLE_SELECTOR_TARGET_NAME} SHARED ${SIMPLE_SELECTOR_SOURCE_FILES})

target_link_libraries(${SIMPLE_SELECTOR_TARGET_NAME}
    cl_add
    cl_concat_xy
    cl_concat_z
    cl_gpu_operation
    cl_lstm
    cl_max_unpooling
    cl_mean
    cl_multiply_add
    cl_padding
    cl_pooling
    cl_prelu
    cl_relu
    cl_reshape
    cl_reshapex4
    cl_resize
    cl_softmax
    cl_softmax1x1
    cl_space_to_depth
    cl_strided_slice
    cl_transpose
    cl_winograd
    common_operations
    common_shape
    absl::memory
)

install(
    TARGETS ${SIMPLE_SELECTOR_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### operation_selector #############################
set(OPERATION_SELECTOR_TARGET_NAME operation_selector)

set(OPERATION_SELECTOR_SOURCE_FILES
    operation_selector.cc
)

add_library(${OPERATION_SELECTOR_TARGET_NAME} SHARED ${OPERATION_SELECTOR_SOURCE_FILES})

target_link_libraries(${OPERATION_SELECTOR_TARGET_NAME}
    convolution_selector
    convolution_transposed_selector
    dw_convolution_selector
    fully_connected_selector
    simple_selectors
    cl_device
    cl_storage_type_util
    cl_tensor_type
    cl_elementwise
    cl_gpu_operation
    common_data_type
    common_shape
    absl::strings
    absl::any
)

install(
    TARGETS ${OPERATION_SELECTOR_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)