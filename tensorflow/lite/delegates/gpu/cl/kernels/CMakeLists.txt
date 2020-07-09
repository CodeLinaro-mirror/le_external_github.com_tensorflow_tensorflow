########################### cl_add #############################
set(CL_ADD_TARGET_NAME cl_add)

set(CL_ADD_SOURCE_FILES
    add.cc
)

add_library(${CL_ADD_TARGET_NAME} SHARED ${CL_ADD_SOURCE_FILES})

target_link_libraries(${CL_ADD_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    common_operations
    absl::strings
)

install(
    TARGETS ${CL_ADD_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_test #############################
set(CL_TEST_TARGET_NAME cl_test)

set(CL_TEST_SOURCE_FILES
    cl_test.cc
)

add_library(${CL_TEST_TARGET_NAME} SHARED ${CL_TEST_SOURCE_FILES})

target_link_libraries(${CL_TEST_TARGET_NAME}
    cl_gpu_operation
    cl_environment
    opencl_wrapper
    common_shape
    gtest
)

install(
    TARGETS ${CL_TEST_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_concat_xy #############################
set(CL_CONCAT_XY_TARGET_NAME cl_concat_xy)

set(CL_CONCAT_XY_SOURCE_FILES
    concat_xy.cc
)

add_library(${CL_CONCAT_XY_TARGET_NAME} SHARED ${CL_CONCAT_XY_SOURCE_FILES})

target_link_libraries(${CL_CONCAT_XY_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_kernel
    common_operations
)

install(
    TARGETS ${CL_CONCAT_XY_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_concat_z #############################
set(CL_CONCAT_Z_TARGET_NAME cl_concat_z)

set(CL_CONCAT_Z_SOURCE_FILES
    concat_z.cc
)

add_library(${CL_CONCAT_Z_TARGET_NAME} SHARED ${CL_CONCAT_Z_SOURCE_FILES})

target_link_libraries(${CL_CONCAT_Z_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_kernel
    cl_tensor
)

install(
    TARGETS ${CL_CONCAT_Z_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_conv_3d #############################
set(CL_CONV_3D_TARGET_NAME cl_conv_3d)

set(CL_CONV_3D_SOURCE_FILES
    conv_3d.cc
)

add_library(${CL_CONV_3D_TARGET_NAME} SHARED ${CL_CONV_3D_SOURCE_FILES})

target_link_libraries(${CL_CONV_3D_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_buffer
    cl_device
    cl_linear_storage
    cl_precision
    cl_tensor
    cl_tensor_type
    cl_util
    common_data_type
    common_operations
    common_shape
    absl::strings
)

install(
    TARGETS ${CL_CONV_3D_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_conv_buffer_1x1 #############################
set(CL_CONV_BUFF_1X1_TARGET_NAME cl_conv_buffer_1x1)

set(CL_CONV_BUFF_1X1_SOURCE_FILES
    conv_buffer_1x1.cc
)

add_library(${CL_CONV_BUFF_1X1_TARGET_NAME} SHARED ${CL_CONV_BUFF_1X1_SOURCE_FILES})

target_link_libraries(${CL_CONV_BUFF_1X1_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_buffer
    cl_device
    cl_kernel
    cl_linear_storage
    cl_precision
    cl_tensor
    cl_tensor_type
    cl_util
    common_data_type
    common_operations
    common_shape
)

install(
    TARGETS ${CL_CONV_BUFF_1X1_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_conv_constants #############################
set(CL_CONV_CONSTANTS_TARGET_NAME cl_conv_constants)

set(CL_CONV_CONSTANTS_SOURCE_FILES
    conv_constants.cc
)

add_library(${CL_CONV_CONSTANTS_TARGET_NAME} SHARED ${CL_CONV_CONSTANTS_SOURCE_FILES})

target_link_libraries(${CL_CONV_CONSTANTS_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_buffer
    cl_linear_storage
    cl_precision
    cl_tensor
    cl_util
    common_data_type
    common_operations
    common_shape
)

install(
    TARGETS ${CL_CONV_CONSTANTS_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_conv_powervr #############################
set(CL_CONV_POWERVR_TARGET_NAME cl_conv_powervr)

set(CL_CONV_POWERVR_SOURCE_FILES
    conv_powervr.cc
)

add_library(${CL_CONV_POWERVR_TARGET_NAME} SHARED ${CL_CONV_POWERVR_SOURCE_FILES})

target_link_libraries(${CL_CONV_POWERVR_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_buffer
    cl_device
    cl_linear_storage
    cl_precision
    cl_tensor
    cl_tensor_type
    cl_util
    common_data_type
    common_operations
    common_shape
)

install(
    TARGETS ${CL_CONV_POWERVR_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_conv_texture #############################
set(CL_CONV_TEXTURE_TARGET_NAME cl_conv_texture)

set(CL_CONV_TEXTURE_SOURCE_FILES
    conv_texture.cc
)

add_library(${CL_CONV_TEXTURE_TARGET_NAME} SHARED ${CL_CONV_TEXTURE_SOURCE_FILES})

target_link_libraries(${CL_CONV_TEXTURE_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_command_queue
    cl_context
    cl_linear_storage
    cl_precision
    cl_tensor
    cl_tensor_type
    cl_texture2d
    cl_util
    common_data_type
    common_operations
    common_shape
    absl::strings
)

install(
    TARGETS ${CL_CONV_TEXTURE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_converter #############################
set(CL_CONVERTER_TARGET_NAME cl_converter)

set(CL_CONVERTER_SOURCE_FILES
    converter.cc
)

add_library(${CL_CONVERTER_TARGET_NAME} SHARED ${CL_CONVERTER_SOURCE_FILES})

target_link_libraries(${CL_CONVERTER_TARGET_NAME}
    cl_kernels_util
    gpu_api
    cl_command_queue
    cl_environment
    cl_precision
    cl_tensor
    cl_tensor_type
    cl_tensor_type_util
)

install(
    TARGETS ${CL_CONVERTER_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_convolution_transposed #############################
set(CL_CONV_TRANSPOSED_TARGET_NAME cl_convolution_transposed)

set(CL_CONV_TRANSPOSED_SOURCE_FILES
    convolution_transposed.cc
)

add_library(${CL_CONV_TRANSPOSED_TARGET_NAME} SHARED ${CL_CONV_TRANSPOSED_SOURCE_FILES})

target_link_libraries(${CL_CONV_TRANSPOSED_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_buffer
    cl_linear_storage
    cl_tensor
    cl_tensor_type
    cl_texture2d
    cl_util
    common_data_type
    common_operations
    common_shape
    absl::strings
)

install(
    TARGETS ${CL_CONV_TRANSPOSED_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_convolution_transposed_3d #############################
set(CL_CONV_TRANSPOSED_3D_TARGET_NAME cl_convolution_transposed_3d)

set(CL_CONV_TRANSPOSED_3D_SOURCE_FILES
    convolution_transposed_3d.cc
)

add_library(${CL_CONV_TRANSPOSED_3D_TARGET_NAME} SHARED ${CL_CONV_TRANSPOSED_3D_SOURCE_FILES})

target_link_libraries(${CL_CONV_TRANSPOSED_3D_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_buffer
    cl_linear_storage
    cl_tensor
    cl_tensor_type
    cl_texture2d
    cl_util
    common_data_type
    common_operations
    common_shape
    absl::strings
)

install(
    TARGETS ${CL_CONV_TRANSPOSED_3D_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_convolution_transposed_3x3 #############################
set(CL_CONV_TRANSPOSED_3X3_TARGET_NAME cl_convolution_transposed_3x3)

set(CL_CONV_TRANSPOSED_3X3_SOURCE_FILES
    convolution_transposed_3x3.cc
)

add_library(${CL_CONV_TRANSPOSED_3X3_TARGET_NAME} SHARED ${CL_CONV_TRANSPOSED_3X3_SOURCE_FILES})

target_link_libraries(${CL_CONV_TRANSPOSED_3X3_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_buffer
    cl_linear_storage
    cl_precision
    cl_tensor
    cl_tensor_type
    cl_util
    common_data_type
    common_operations
    common_shape
)

install(
    TARGETS ${CL_CONV_TRANSPOSED_3X3_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_convolution_transposed_3x3_thin #############################
set(CL_CONV_TRANSPOSED_3X3_THIN_TARGET_NAME cl_convolution_transposed_3x3_thin)

set(CL_CONV_TRANSPOSED_3X3_THIN_SOURCE_FILES
    convolution_transposed_3x3_thin.cc
)

add_library(${CL_CONV_TRANSPOSED_3X3_THIN_TARGET_NAME} SHARED ${CL_CONV_TRANSPOSED_3X3_THIN_SOURCE_FILES})

target_link_libraries(${CL_CONV_TRANSPOSED_3X3_THIN_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_buffer
    cl_linear_storage
    cl_tensor
    cl_util
    common_data_type
    common_operations
    common_shape
)

install(
    TARGETS ${CL_CONV_TRANSPOSED_3X3_THIN_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_convolution_transposed_4x4 #############################
set(CL_CONV_TRANSPOSED_4X4_TARGET_NAME cl_convolution_transposed_4x4)

set(CL_CONV_TRANSPOSED_4X4_SOURCE_FILES
    convolution_transposed_4x4.cc
)

add_library(${CL_CONV_TRANSPOSED_4X4_TARGET_NAME} SHARED ${CL_CONV_TRANSPOSED_4X4_SOURCE_FILES})

target_link_libraries(${CL_CONV_TRANSPOSED_4X4_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_buffer
    cl_linear_storage
    cl_precision
    cl_tensor
    cl_tensor_type
    cl_util
    common_data_type
    common_operations
    common_shape
)

install(
    TARGETS ${CL_CONV_TRANSPOSED_4X4_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_convolution_transposed_thin #############################
set(CL_CONV_TRANSPOSED_THIN_TARGET_NAME cl_convolution_transposed_thin)

set(CL_CONV_TRANSPOSED_THIN_SOURCE_FILES
    convolution_transposed_thin.cc
)

add_library(${CL_CONV_TRANSPOSED_THIN_TARGET_NAME} SHARED ${CL_CONV_TRANSPOSED_THIN_SOURCE_FILES})

target_link_libraries(${CL_CONV_TRANSPOSED_THIN_TARGET_NAME}
    cl_flt_type
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_buffer
    cl_precision
    cl_tensor
    cl_texture2d
    cl_util
    common_data_type
    common_operations
    common_shape
)

install(
    TARGETS ${CL_CONV_TRANSPOSED_THIN_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_depth_wise_conv #############################
set(CL_DEPTH_WISE_CONV_TARGET_NAME cl_depth_wise_conv)

set(CL_DEPTH_WISE_CONV_SOURCE_FILES
    depth_wise_conv.cc
)

add_library(${CL_DEPTH_WISE_CONV_TARGET_NAME} SHARED ${CL_DEPTH_WISE_CONV_SOURCE_FILES})

target_link_libraries(${CL_DEPTH_WISE_CONV_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_buffer
    cl_device
    cl_linear_storage
    cl_tensor
    cl_texture2d
    cl_util
    common_data_type
    common_operations
    common_shape
)

install(
    TARGETS ${CL_DEPTH_WISE_CONV_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_depth_wise_conv_3d #############################
set(CL_DEPTH_WISE_CONV_3D_TARGET_NAME cl_depth_wise_conv_3d)

set(CL_DEPTH_WISE_CONV_3D_SOURCE_FILES
    depth_wise_conv_3d.cc
)

add_library(${CL_DEPTH_WISE_CONV_3D_TARGET_NAME} SHARED ${CL_DEPTH_WISE_CONV_3D_SOURCE_FILES})

target_link_libraries(${CL_DEPTH_WISE_CONV_3D_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_buffer
    cl_device
    cl_linear_storage
    cl_tensor
    cl_texture2d
    cl_util
    common_data_type
    common_operations
    common_shape
)

install(
    TARGETS ${CL_DEPTH_WISE_CONV_3D_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_depth_wise_conv_3x3 #############################
set(CL_DEPTH_WISE_CONV_3X3_TARGET_NAME cl_depth_wise_conv_3x3)

set(CL_DEPTH_WISE_CONV_3X3_SOURCE_FILES
    depth_wise_conv_3x3.cc
)

add_library(${CL_DEPTH_WISE_CONV_3X3_TARGET_NAME} SHARED ${CL_DEPTH_WISE_CONV_3X3_SOURCE_FILES})

target_link_libraries(${CL_DEPTH_WISE_CONV_3X3_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_buffer
    cl_precision
    cl_tensor
    cl_texture2d
    cl_util
    common_data_type
    common_operations
    common_shape
)

install(
    TARGETS ${CL_DEPTH_WISE_CONV_3X3_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_elementwise #############################
set(CL_ELEMENTWISE_TARGET_NAME cl_elementwise)

set(CL_ELEMENTWISE_SOURCE_FILES
    elementwise.cc
)

add_library(${CL_ELEMENTWISE_TARGET_NAME} SHARED ${CL_ELEMENTWISE_SOURCE_FILES})

target_link_libraries(${CL_ELEMENTWISE_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    common_operations
    absl::strings
)

install(
    TARGETS ${CL_ELEMENTWISE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_flt_type #############################
set(CL_FTL_TYPE_TARGET_NAME cl_flt_type)

set(CL_FTL_TYPE_SOURCE_FILES
    flt_type.cc
)

add_library(${CL_FTL_TYPE_TARGET_NAME} SHARED ${CL_FTL_TYPE_SOURCE_FILES})

target_link_libraries(${CL_FTL_TYPE_TARGET_NAME}
    cl_precision
    absl::strings
)

install(
    TARGETS ${CL_FTL_TYPE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_fully_connected #############################
set(CL_FULLY_CONNECTED_TARGET_NAME cl_fully_connected)

set(CL_FULLY_CONNECTED_SOURCE_FILES
    fully_connected.cc
)

add_library(${CL_FULLY_CONNECTED_TARGET_NAME} SHARED ${CL_FULLY_CONNECTED_SOURCE_FILES})

target_link_libraries(${CL_FULLY_CONNECTED_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_buffer
    cl_linear_storage
    cl_tensor
    cl_util
    common_data_type
    common_operations
    common_shape
)

install(
    TARGETS ${CL_FULLY_CONNECTED_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_gpu_operation #############################
set(CL_GPU_OPERATION_TARGET_NAME cl_gpu_operation)

set(CL_GPU_OPERATION_SOURCE_FILES
    gpu_operation.cc
)

add_library(${CL_GPU_OPERATION_TARGET_NAME} SHARED ${CL_GPU_OPERATION_SOURCE_FILES})

target_link_libraries(${CL_GPU_OPERATION_TARGET_NAME}
    cl_command_queue
    cl_device
    cl_kernels_util
    cl_work_group_picking
    cl_context
    cl_device
    cl_precision
    cl_program_cache
    cl_tensor
    cl_tensor_type
)

install(
    TARGETS ${CL_GPU_OPERATION_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_lstm #############################
set(CL_LSTM_TARGET_NAME cl_lstm)

set(CL_LSTM_SOURCE_FILES
    lstm.cc
)

add_library(${CL_LSTM_TARGET_NAME} SHARED ${CL_LSTM_SOURCE_FILES})

target_link_libraries(${CL_LSTM_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_precision
    common_operations
)

install(
    TARGETS ${CL_LSTM_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_max_unpooling #############################
set(CL_MAX_UNPOOLING_TARGET_NAME cl_max_unpooling)

set(CL_MAX_UNPOOLING_SOURCE_FILES
    max_unpooling.cc
)

add_library(${CL_MAX_UNPOOLING_TARGET_NAME} SHARED ${CL_MAX_UNPOOLING_SOURCE_FILES})

target_link_libraries(${CL_MAX_UNPOOLING_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    common_operations
)

install(
    TARGETS ${CL_MAX_UNPOOLING_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_mean #############################
set(CL_MEAN_TARGET_NAME cl_mean)

set(CL_MEAN_SOURCE_FILES
    mean.cc
)

add_library(${CL_MEAN_TARGET_NAME} SHARED ${CL_MEAN_SOURCE_FILES})

target_link_libraries(${CL_MEAN_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_kernel
    cl_precision
    cl_tensor
)

install(
    TARGETS ${CL_MEAN_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_multiply_add #############################
set(CL_MULTIPLY_ADD_TARGET_NAME cl_multiply_add)

set(CL_MULTIPLY_ADD_SOURCE_FILES
    multiply_add.cc
)

add_library(${CL_MULTIPLY_ADD_TARGET_NAME} SHARED ${CL_MULTIPLY_ADD_SOURCE_FILES})

target_link_libraries(${CL_MULTIPLY_ADD_TARGET_NAME}
    cl_flt_type
    cl_gpu_operation
    cl_kernels_util
    cl_context
    cl_linear_storage
    cl_precision
    common_data_type
    common_operations
    common_shape
    absl::variant
)

install(
    TARGETS ${CL_MULTIPLY_ADD_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_padding #############################
set(CL_PADDING_TARGET_NAME cl_padding)

set(CL_PADDING_SOURCE_FILES
    padding.cc
)

add_library(${CL_PADDING_TARGET_NAME} SHARED ${CL_PADDING_SOURCE_FILES})

target_link_libraries(${CL_PADDING_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    common_operations
)

install(
    TARGETS ${CL_PADDING_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_pooling #############################
set(CL_POOLING_TARGET_NAME cl_pooling)

set(CL_POOLING_SOURCE_FILES
    pooling.cc
)

add_library(${CL_POOLING_TARGET_NAME} SHARED ${CL_POOLING_SOURCE_FILES})

target_link_libraries(${CL_POOLING_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_kernel
    cl_tensor
    common_operations
)

install(
    TARGETS ${CL_POOLING_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_prelu #############################
set(CL_PRELU_TARGET_NAME cl_prelu)

set(CL_PRELU_SOURCE_FILES
    prelu.cc
)

add_library(${CL_PRELU_TARGET_NAME} SHARED ${CL_PRELU_SOURCE_FILES})

target_link_libraries(${CL_PRELU_TARGET_NAME}
    cl_flt_type
    cl_gpu_operation
    cl_kernels_util
    cl_context
    cl_kernel
    cl_linear_storage
    common_data_type
    common_operations
    absl::strings
    absl::variant
)

install(
    TARGETS ${CL_PRELU_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_relu #############################
set(CL_RELU_TARGET_NAME cl_relu)

set(CL_RELU_SOURCE_FILES
    relu.cc
)

add_library(${CL_RELU_TARGET_NAME} SHARED ${CL_RELU_SOURCE_FILES})

target_link_libraries(${CL_RELU_TARGET_NAME}
    cl_flt_type
    cl_gpu_operation
    cl_precision
    common_operations
    absl::strings
)

install(
    TARGETS ${CL_RELU_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_reshape #############################
set(CL_RESHAPE_TARGET_NAME cl_reshape)

set(CL_RESHAPE_SOURCE_FILES
    reshape.cc
)

add_library(${CL_RESHAPE_TARGET_NAME} SHARED ${CL_RESHAPE_SOURCE_FILES})

target_link_libraries(${CL_RESHAPE_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_kernel
)

install(
    TARGETS ${CL_RESHAPE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_reshapex4 #############################
set(CL_RESHAPEX4_TARGET_NAME cl_reshapex4)

set(CL_RESHAPEX4_SOURCE_FILES
    reshapex4.cc
)

add_library(${CL_RESHAPEX4_TARGET_NAME} SHARED ${CL_RESHAPEX4_SOURCE_FILES})

target_link_libraries(${CL_RESHAPEX4_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_command_queue
    cl_kernel
)

install(
    TARGETS ${CL_RESHAPEX4_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_softmax #############################
set(CL_SOFTMAX_TARGET_NAME cl_softmax)

set(CL_SOFTMAX_SOURCE_FILES
    softmax.cc
)

add_library(${CL_SOFTMAX_TARGET_NAME} SHARED ${CL_SOFTMAX_SOURCE_FILES})

target_link_libraries(${CL_SOFTMAX_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_kernel
    cl_precision
    cl_tensor
)

install(
    TARGETS ${CL_SOFTMAX_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_softmax1x1 #############################
set(CL_SOFTMAX1X1_TARGET_NAME cl_softmax1x1)

set(CL_SOFTMAX1X1_SOURCE_FILES
    softmax1x1.cc
)

add_library(${CL_SOFTMAX1X1_TARGET_NAME} SHARED ${CL_SOFTMAX1X1_SOURCE_FILES})

target_link_libraries(${CL_SOFTMAX1X1_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_kernel
    cl_precision
    cl_tensor
)

install(
    TARGETS ${CL_SOFTMAX1X1_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_space_to_depth #############################
set(CL_SPACE_TO_DEPTH_TARGET_NAME cl_space_to_depth)

set(CL_SPACE_TO_DEPTH_SOURCE_FILES
    space_to_depth.cc
)

add_library(${CL_SPACE_TO_DEPTH_TARGET_NAME} SHARED ${CL_SPACE_TO_DEPTH_SOURCE_FILES})

target_link_libraries(${CL_SPACE_TO_DEPTH_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_kernel
    common_operations
)

install(
    TARGETS ${CL_SPACE_TO_DEPTH_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_strided_slice #############################
set(CL_STRIDED_SLICE_TARGET_NAME cl_strided_slice)

set(CL_STRIDED_SLICE_SOURCE_FILES
    strided_slice.cc
)

add_library(${CL_STRIDED_SLICE_TARGET_NAME} SHARED ${CL_STRIDED_SLICE_SOURCE_FILES})

target_link_libraries(${CL_STRIDED_SLICE_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    common_operations
)

install(
    TARGETS ${CL_STRIDED_SLICE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_transpose #############################
set(CL_TRANSPOSE_TARGET_NAME cl_transpose)

set(CL_TRANSPOSE_SOURCE_FILES
    transpose.cc
)

add_library(${CL_TRANSPOSE_TARGET_NAME} SHARED ${CL_TRANSPOSE_SOURCE_FILES})

target_link_libraries(${CL_TRANSPOSE_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    common_operations
)

install(
    TARGETS ${CL_TRANSPOSE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_resize #############################
set(CL_RESIZE_TARGET_NAME cl_resize)

set(CL_RESIZE_SOURCE_FILES
    resize.cc
)

add_library(${CL_RESIZE_TARGET_NAME} SHARED ${CL_RESIZE_SOURCE_FILES})

target_link_libraries(${CL_RESIZE_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_tensor_type
    common_operations
)

install(
    TARGETS ${CL_RESIZE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_kernels_util #############################
set(CL_UTIL_TARGET_NAME cl_kernels_util)

set(CL_UTIL_SOURCE_FILES
    util.cc
)

add_library(${CL_UTIL_TARGET_NAME} SHARED ${CL_UTIL_SOURCE_FILES})

target_link_libraries(${CL_UTIL_TARGET_NAME}
    cl_device
    cl_precision
    cl_tensor_type
    common_data_type
    common_shape
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

########################### cl_winograd #############################
set(CL_WINOGRAD_TARGET_NAME cl_winograd)

set(CL_WINOGRAD_SOURCE_FILES
    winograd.cc
)

add_library(${CL_WINOGRAD_TARGET_NAME} SHARED ${CL_WINOGRAD_SOURCE_FILES})

target_link_libraries(${CL_WINOGRAD_TARGET_NAME}
    cl_gpu_operation
    cl_kernels_util
    cl_work_group_picking
    cl_device
    cl_kernel
    cl_linear_storage
    cl_precision
    cl_tensor
    common_data_type
    common_operations
    common_shape
    absl::str_format
)

install(
    TARGETS ${CL_WINOGRAD_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### cl_work_group_picking #############################
set(CL_WORK_GROUP_PICKING_TARGET_NAME cl_work_group_picking)

set(CL_WORK_GROUP_PICKING_SOURCE_FILES
    work_group_picking.cc
)

add_library(${CL_WORK_GROUP_PICKING_TARGET_NAME} SHARED ${CL_WORK_GROUP_PICKING_SOURCE_FILES})

target_link_libraries(${CL_WORK_GROUP_PICKING_TARGET_NAME}
    cl_command_queue
    cl_device
    cl_kernel
    common_workgroup_selection
)

install(
    TARGETS ${CL_WORK_GROUP_PICKING_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)