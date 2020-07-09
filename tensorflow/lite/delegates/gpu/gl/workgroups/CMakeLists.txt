########################### workgroups_calculator #############################
set(WGR_TARGET_NAME workgroups_calculator)

set(WGR_SOURCE_FILES
    calculator.cc
)

add_library(${WGR_TARGET_NAME} SHARED ${WGR_SOURCE_FILES})

target_link_libraries(${WGR_TARGET_NAME}
    common_gpu_info
)

install(
    TARGETS ${WGR_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### workgroups_default_calculator #############################
set(WGR_DEFAULT_TARGET_NAME workgroups_default_calculator)

set(WGR_DEFAULT_SOURCE_FILES
    default_calculator.cc
)

add_library(${WGR_DEFAULT_TARGET_NAME} SHARED ${WGR_DEFAULT_SOURCE_FILES})

target_link_libraries(${WGR_DEFAULT_TARGET_NAME}
    workgroups_calculator
    common_gpu_info
)

install(
    TARGETS ${WGR_DEFAULT_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### workgroups_calculator_from_metadata #############################
set(WGR_BEST_EFFORT_TARGET_NAME workgroups_calculator_from_metadata)

set(WGR_BEST_EFFORT_SOURCE_FILES
    calculator_from_metadata.cc
)

add_library(${WGR_BEST_EFFORT_TARGET_NAME} SHARED ${WGR_BEST_EFFORT_SOURCE_FILES})

add_dependencies(${WGR_BEST_EFFORT_TARGET_NAME}
    ${COMMON_CC_FBS_TARGET}
    ${METADATA_CC_FBS_TARGET}
    ${WORKGROUPS_CC_FBS_TARGET}
)

target_include_directories(${WGR_BEST_EFFORT_TARGET_NAME} PRIVATE ${CMAKE_BINARY_DIR})

target_link_libraries(${WGR_BEST_EFFORT_TARGET_NAME}
    common_gpu_info
    workgroups_default_calculator
    workgroups_calculator
    absl::memory
    flatbuffers
)

install(
    TARGETS ${WGR_BEST_EFFORT_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### workgroups_best_effort_calculator #############################
set(WGR_BEST_EFFORT_TARGET_NAME workgroups_best_effort_calculator)

set(WGR_BEST_EFFORT_SOURCE_FILES
    best_effort_calculator.cc
)

add_library(${WGR_BEST_EFFORT_TARGET_NAME} SHARED ${WGR_BEST_EFFORT_SOURCE_FILES})

target_link_libraries(${WGR_BEST_EFFORT_TARGET_NAME}
    workgroups_calculator
    workgroups_default_calculator
    workgroups_calculator_from_metadata
    common_gpu_info
)

install(
    TARGETS ${WGR_BEST_EFFORT_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### workgroups_ideal_workgroup_picker #############################
set(WGR_IDEAL_WORKGROUP_PICKER_TARGET_NAME workgroups_ideal_workgroup_picker)

set(WGR_IDEAL_WORKGROUP_PICKER_SOURCE_FILES
    ideal_workgroup_picker.cc
)

add_library(${WGR_IDEAL_WORKGROUP_PICKER_TARGET_NAME} SHARED ${WGR_IDEAL_WORKGROUP_PICKER_SOURCE_FILES})

target_link_libraries(${WGR_IDEAL_WORKGROUP_PICKER_TARGET_NAME}
    workgroups_calculator
    common_gpu_info
    common_operations
    common_shape
)

install(
    TARGETS ${WGR_IDEAL_WORKGROUP_PICKER_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)