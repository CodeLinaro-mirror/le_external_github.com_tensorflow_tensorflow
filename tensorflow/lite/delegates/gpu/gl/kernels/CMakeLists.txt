########################### gl_add #############################
set(GL_ADD_TARGET_NAME gl_add)

set(GL_ADD_SOURCE_FILES
    add.cc
)

add_library(${GL_ADD_TARGET_NAME} SHARED ${GL_ADD_SOURCE_FILES})

target_link_libraries(${GL_ADD_TARGET_NAME}
    common_operations
    absl::memory
    absl::strings
)

install(
    TARGETS ${GL_ADD_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_concat #############################
set(GL_CONCAT_TARGET_NAME gl_concat)

set(GL_CONCAT_SOURCE_FILES
    concat.cc
)

add_library(${GL_CONCAT_TARGET_NAME} SHARED ${GL_CONCAT_SOURCE_FILES})

target_link_libraries(${GL_CONCAT_TARGET_NAME}
    common_operations
    absl::variant
    absl::memory
)

install(
    TARGETS ${GL_CONCAT_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_conv #############################
set(GL_CONV_TARGET_NAME gl_conv)

set(GL_CONV_SOURCE_FILES
    conv.cc
)

add_library(${GL_CONV_TARGET_NAME} SHARED ${GL_CONV_SOURCE_FILES})

target_link_libraries(${GL_CONV_TARGET_NAME}
    common_convert
    common_operations
    common_shape
    workgroups_ideal_workgroup_picker
    absl::memory
    absl::strings
    absl::variant
)

install(
    TARGETS ${GL_CONV_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_custom_registry #############################
set(GL_CUSTOM_REGISTRY_TARGET_NAME gl_custom_registry)

set(GL_CUSTOM_REGISTRY_SOURCE_FILES
    custom_registry.cc
)

add_library(${GL_CUSTOM_REGISTRY_TARGET_NAME} SHARED ${GL_CUSTOM_REGISTRY_SOURCE_FILES})

install(
    TARGETS ${GL_CUSTOM_REGISTRY_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_depthwise_conv #############################
set(GL_DEPTHWISE_CONV_TARGET_NAME gl_depthwise_conv)

set(GL_DEPTHWISE_CONV_SOURCE_FILES
    depthwise_conv.cc
)

add_library(${GL_DEPTHWISE_CONV_TARGET_NAME} SHARED ${GL_DEPTHWISE_CONV_SOURCE_FILES})

target_link_libraries(${GL_DEPTHWISE_CONV_TARGET_NAME}
    common_convert
    common_operations
    common_shape
    workgroups_ideal_workgroup_picker
    absl::memory
    absl::variant
)

install(
    TARGETS ${GL_DEPTHWISE_CONV_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_elementwise #############################
set(GL_ELEMENTWISE_TARGET_NAME gl_elementwise)

set(GL_ELEMENTWISE_SOURCE_FILES
    elementwise.cc
)

add_library(${GL_ELEMENTWISE_TARGET_NAME} SHARED ${GL_ELEMENTWISE_SOURCE_FILES})

target_link_libraries(${GL_ELEMENTWISE_TARGET_NAME}
    common_operations
    absl::memory
)

install(
    TARGETS ${GL_ELEMENTWISE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_fully_connected #############################
set(GL_FULLY_CONNECTED_TARGET_NAME gl_fully_connected)

set(GL_FULLY_CONNECTED_SOURCE_FILES
    fully_connected.cc
)

add_library(${GL_FULLY_CONNECTED_TARGET_NAME} SHARED ${GL_FULLY_CONNECTED_SOURCE_FILES})

target_link_libraries(${GL_FULLY_CONNECTED_TARGET_NAME}
    common_convert
    common_operations
    absl::variant
    absl::memory
)

install(
    TARGETS ${GL_FULLY_CONNECTED_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_lstm #############################
set(GL_LSTM_TARGET_NAME gl_lstm)

set(GL_LSTM_SOURCE_FILES
    lstm.cc
)

add_library(${GL_LSTM_TARGET_NAME} SHARED ${GL_LSTM_SOURCE_FILES})

target_link_libraries(${GL_LSTM_TARGET_NAME}
    common_operations
    absl::memory
)

install(
    TARGETS ${GL_LSTM_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_max_unpooling #############################
set(GL_MAX_UNPOOLING_TARGET_NAME gl_max_unpooling)

set(GL_MAX_UNPOOLING_SOURCE_FILES
    max_unpooling.cc
)

add_library(${GL_MAX_UNPOOLING_TARGET_NAME} SHARED ${GL_MAX_UNPOOLING_SOURCE_FILES})

target_link_libraries(${GL_MAX_UNPOOLING_TARGET_NAME}
    common_operations
    absl::variant
    absl::memory
)

install(
    TARGETS ${GL_MAX_UNPOOLING_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_mean #############################
set(GL_MEAN_TARGET_NAME gl_mean)

set(GL_MEAN_SOURCE_FILES
    mean.cc
)

add_library(${GL_MEAN_TARGET_NAME} SHARED ${GL_MEAN_SOURCE_FILES})

target_link_libraries(${GL_MEAN_TARGET_NAME}
    common_operations
    absl::memory
    absl::strings
)

install(
    TARGETS ${GL_MEAN_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_mul #############################
set(GL_MUL_TARGET_NAME gl_mul)

set(GL_MUL_SOURCE_FILES
    mul.cc
)

add_library(${GL_MUL_TARGET_NAME} SHARED ${GL_MUL_SOURCE_FILES})

target_link_libraries(${GL_MUL_TARGET_NAME}
    common_operations
    absl::memory
    absl::strings
)

install(
    TARGETS ${GL_MUL_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_pad #############################
set(GL_PAD_TARGET_NAME gl_pad)

set(GL_PAD_SOURCE_FILES
    pad.cc
)

add_library(${GL_PAD_TARGET_NAME} SHARED ${GL_PAD_SOURCE_FILES})

target_link_libraries(${GL_PAD_TARGET_NAME}
    common_operations
    absl::variant
    absl::memory
)

install(
    TARGETS ${GL_PAD_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_pooling #############################
set(GL_POOLING_TARGET_NAME gl_pooling)

set(GL_POOLING_SOURCE_FILES
    pooling.cc
)

add_library(${GL_POOLING_TARGET_NAME} SHARED ${GL_POOLING_SOURCE_FILES})

target_link_libraries(${GL_POOLING_TARGET_NAME}
    common_operations
    absl::variant
    absl::memory
)

install(
    TARGETS ${GL_POOLING_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_prelu #############################
set(GL_PRELU_TARGET_NAME gl_prelu)

set(GL_PRELU_SOURCE_FILES
    prelu.cc
)

add_library(${GL_PRELU_TARGET_NAME} SHARED ${GL_PRELU_SOURCE_FILES})

target_link_libraries(${GL_PRELU_TARGET_NAME}
    common_convert
    common_operations
    common_shape
    absl::memory
)

install(
    TARGETS ${GL_PRELU_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_relu #############################
set(GL_RELU_TARGET_NAME gl_relu)

set(GL_RELU_SOURCE_FILES
    relu.cc
)

add_library(${GL_RELU_TARGET_NAME} SHARED ${GL_RELU_SOURCE_FILES})

target_link_libraries(${GL_RELU_TARGET_NAME}
    common_operations
    absl::variant
    absl::memory
)

install(
    TARGETS ${GL_RELU_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_reshape #############################
set(GL_RESHAPE_TARGET_NAME gl_reshape)

set(GL_RESHAPE_SOURCE_FILES
    reshape.cc
)

add_library(${GL_RESHAPE_TARGET_NAME} SHARED ${GL_RESHAPE_SOURCE_FILES})

target_link_libraries(${GL_RESHAPE_TARGET_NAME}
    common_operations
    absl::memory
)

install(
    TARGETS ${GL_RESHAPE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_slice #############################
set(GL_SLICE_TARGET_NAME gl_slice)

set(GL_SLICE_SOURCE_FILES
    slice.cc
)

add_library(${GL_SLICE_TARGET_NAME} SHARED ${GL_SLICE_SOURCE_FILES})

target_link_libraries(${GL_SLICE_TARGET_NAME}
    common_operations
    absl::variant
    absl::memory
)

install(
    TARGETS ${GL_SLICE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_softmax #############################
set(GL_SOFTMAX_TARGET_NAME gl_softmax)

set(GL_SOFTMAX_SOURCE_FILES
    softmax.cc
)

add_library(${GL_SOFTMAX_TARGET_NAME} SHARED ${GL_SOFTMAX_SOURCE_FILES})

target_link_libraries(${GL_SOFTMAX_TARGET_NAME}
    common_operations
    common_shape
    absl::variant
    absl::memory
)

install(
    TARGETS ${GL_SOFTMAX_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_space_to_depth #############################
set(GL_SPACE_TO_DEPTH_TARGET_NAME gl_space_to_depth)

set(GL_SPACE_TO_DEPTH_SOURCE_FILES
    space_to_depth.cc
)

add_library(${GL_SPACE_TO_DEPTH_TARGET_NAME} SHARED ${GL_SPACE_TO_DEPTH_SOURCE_FILES})

target_link_libraries(${GL_SPACE_TO_DEPTH_TARGET_NAME}
    common_operations
    absl::memory
    absl::any
)

install(
    TARGETS ${GL_SPACE_TO_DEPTH_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_transpose_conv #############################
set(GL_TRANSPOSE_CONV_TARGET_NAME gl_transpose_conv)

set(GL_TRANSPOSE_CONV_SOURCE_FILES
    transpose_conv.cc
)

add_library(${GL_TRANSPOSE_CONV_TARGET_NAME} SHARED ${GL_TRANSPOSE_CONV_SOURCE_FILES})

target_link_libraries(${GL_TRANSPOSE_CONV_TARGET_NAME}
    common_convert
    common_operations
    common_shape
    absl::variant
    absl::memory
)

install(
    TARGETS ${GL_TRANSPOSE_CONV_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_resize #############################
set(GL_RESIZE_TARGET_NAME gl_resize)

set(GL_RESIZE_SOURCE_FILES
    resize.cc
)

add_library(${GL_RESIZE_TARGET_NAME} SHARED ${GL_RESIZE_SOURCE_FILES})

target_link_libraries(${GL_RESIZE_TARGET_NAME}
    common_operations
    absl::variant
    absl::memory
)

install(
    TARGETS ${GL_RESIZE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_converter #############################
set(GL_CONVERTER_TARGET_NAME gl_converter)

set(GL_CONVERTER_SOURCE_FILES
    converter.cc
)

add_library(${GL_CONVERTER_TARGET_NAME} SHARED ${GL_CONVERTER_SOURCE_FILES})

target_link_libraries(${GL_CONVERTER_TARGET_NAME}
    gpu_api
    common_shape
    gl_command_queue
    gl_buffer
    gl_program
    gl_shader
    absl::strings
    absl::span
)

install(
    TARGETS ${GL_CONVERTER_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### gl_registry #############################
set(GL_REGISTRY_TARGET_NAME gl_registry)

set(GL_REGISTRY_SOURCE_FILES
    registry.cc
)

add_library(${GL_REGISTRY_TARGET_NAME} SHARED ${GL_REGISTRY_SOURCE_FILES})

target_link_libraries(${GL_REGISTRY_TARGET_NAME}
    gl_add
    gl_concat
    gl_conv
    gl_depthwise_conv
    gl_elementwise
    gl_fully_connected
    gl_lstm
    gl_mul
    gl_pad
    gl_pooling
    gl_prelu
    gl_relu
    gl_mean
    gl_reshape
    gl_resize
    gl_slice
    gl_softmax
    gl_space_to_depth
    gl_transpose_conv
    gl_max_unpooling
    gl_custom_registry
    common_operations
    absl::memory
    absl::strings
)

install(
    TARGETS ${GL_REGISTRY_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)