########################### common_convert #############################
set(COMMON_CONVERT_TARGET_NAME common_convert)

set(COMMON_CONVERT_SOURCE_FILES
    convert.cc
)

add_library(${COMMON_CONVERT_TARGET_NAME} SHARED ${COMMON_CONVERT_SOURCE_FILES})

target_link_libraries(${COMMON_CONVERT_TARGET_NAME}
    common_shape
    common_data_type
    absl::strings
    absl::span
)

install(
    TARGETS ${COMMON_CONVERT_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### common_custom_parsers #############################
set(COMMON_CUSTOM_PARSERS_TARGET_NAME common_custom_parsers)

set(COMMON_CUSTOM_PARSERS_SOURCE_FILES
    custom_parsers.cc
)

add_library(${COMMON_CUSTOM_PARSERS_TARGET_NAME} SHARED ${COMMON_CUSTOM_PARSERS_SOURCE_FILES})

target_link_libraries(${COMMON_CUSTOM_PARSERS_TARGET_NAME} ${TFLITE_TARGET_NAME}
    common_shape
    flatbuffers
    absl::strings
    absl::any
)

install(
    TARGETS ${COMMON_CUSTOM_PARSERS_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### common_gpu_info #############################
set(COMMON_GPU_INFO_TARGET_NAME common_gpu_info)

set(COMMON_GPU_INFO_SOURCE_FILES
    gpu_info.cc
)

add_library(${COMMON_GPU_INFO_TARGET_NAME} SHARED ${COMMON_GPU_INFO_SOURCE_FILES})

target_link_libraries(${COMMON_GPU_INFO_TARGET_NAME}
    absl::strings
)

install(
    TARGETS ${COMMON_GPU_INFO_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### common_data_type #############################
set(COMMON_DATA_TYPE_TARGET_NAME common_data_type)

set(COMMON_DATA_TYPE_SOURCE_FILES
    data_type.cc
)

add_library(${COMMON_DATA_TYPE_TARGET_NAME} SHARED ${COMMON_DATA_TYPE_SOURCE_FILES})

install(
    TARGETS ${COMMON_DATA_TYPE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### common_memory_management #############################
set(COMMON_MEMORY_MANAGEMENT_TARGET_NAME common_memory_management)

set(COMMON_MEMORY_MANAGEMENT_SOURCE_FILES
    memory_management.cc
    memory_management/greedy_by_breadth_assignment.cc
    memory_management/greedy_by_size_assignment.cc
    memory_management/internal.cc
    memory_management/min_cost_flow_assignment.cc
    memory_management/types.cc
)

add_library(${COMMON_MEMORY_MANAGEMENT_TARGET_NAME} SHARED ${COMMON_MEMORY_MANAGEMENT_SOURCE_FILES})

target_link_libraries(${COMMON_MEMORY_MANAGEMENT_TARGET_NAME}
    common_shape
    absl::flat_hash_map
    absl::memory
)

install(
    TARGETS ${COMMON_MEMORY_MANAGEMENT_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### common_model_builder #############################
set(COMMON_MODEL_BUILDER_TARGET_NAME common_model_builder)

set(COMMON_MODEL_BUILDER_SOURCE_FILES
    model_builder.cc
)

add_library(${COMMON_MODEL_BUILDER_TARGET_NAME} SHARED ${COMMON_MODEL_BUILDER_SOURCE_FILES})

target_link_libraries(${COMMON_MODEL_BUILDER_TARGET_NAME} ${TFLITE_TARGET_NAME}
    common_custom_parsers
    common_data_type
    common_operations
    common_shape
    general_transformations
    absl::memory
    absl::strings
    absl::str_format
)

install(
    TARGETS ${COMMON_MODEL_BUILDER_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### common_model_transformer #############################
set(COMMON_MODEL_TRANSFORMER_TARGET_NAME common_model_transformer)

set(COMMON_MODEL_TRANSFORMER_SOURCE_FILES
    model_transformer.cc
)

add_library(${COMMON_MODEL_TRANSFORMER_TARGET_NAME} SHARED ${COMMON_MODEL_TRANSFORMER_SOURCE_FILES})

target_link_libraries(${COMMON_MODEL_TRANSFORMER_TARGET_NAME}
    absl::strings
    absl::memory
    absl::any
)

install(
    TARGETS ${COMMON_MODEL_TRANSFORMER_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### common_operations #############################
set(COMMON_OPERATIONS_TARGET_NAME common_operations)

set(COMMON_OPERATIONS_SOURCE_FILES
    operations.cc
)

add_library(${COMMON_OPERATIONS_TARGET_NAME} SHARED ${COMMON_OPERATIONS_SOURCE_FILES})

target_link_libraries(${COMMON_OPERATIONS_TARGET_NAME}
    common_data_type
    common_shape
    absl::memory
    absl::any
    absl::variant
)

install(
    TARGETS ${COMMON_OPERATIONS_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### common_shape #############################
set(COMMON_SHAPE_TARGET_NAME common_shape)

set(COMMON_SHAPE_SOURCE_FILES
    shape.cc
)

add_library(${COMMON_SHAPE_TARGET_NAME} SHARED ${COMMON_SHAPE_SOURCE_FILES})

target_link_libraries(${COMMON_SHAPE_TARGET_NAME}
    absl::hash
    absl::strings
)

install(
    TARGETS ${COMMON_SHAPE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### common_workgroup_selection #############################
set(COMMON_WORKGROUP_SELECTION_TARGET_NAME common_workgroup_selection)

set(COMMON_WORKGROUP_SELECTION_SOURCE_FILES
    workgroup_selection.cc
)

add_library(${COMMON_WORKGROUP_SELECTION_TARGET_NAME} SHARED ${COMMON_WORKGROUP_SELECTION_SOURCE_FILES})

install(
    TARGETS ${COMMON_WORKGROUP_SELECTION_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

add_subdirectory(transformations)