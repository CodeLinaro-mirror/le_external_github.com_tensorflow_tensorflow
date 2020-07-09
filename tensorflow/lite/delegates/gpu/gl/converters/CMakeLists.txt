########################### bhwc_to_phwc4 #############################
set(BHWC_TO_PHWC4_TARGET_NAME bhwc_to_phwc4)

set(BHWC_TO_PHWC4_SOURCE_FILES
    bhwc_to_phwc4.cc
)

add_library(${BHWC_TO_PHWC4_TARGET_NAME} SHARED ${BHWC_TO_PHWC4_SOURCE_FILES})

target_link_libraries(${BHWC_TO_PHWC4_TARGET_NAME}
    common_shape
    gl_command_queue
    gl_buffer
    gl_program
    gl_shader
)

install(
    TARGETS ${BHWC_TO_PHWC4_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### phwc4_to_bhwc #############################
set(PHWC4_TO_BHWC_TARGET_NAME phwc4_to_bhwc)

set(PHWC4_TO_BHWC_SOURCE_FILES
    phwc4_to_bhwc.cc
)

add_library(${PHWC4_TO_BHWC_TARGET_NAME} SHARED ${PHWC4_TO_BHWC_SOURCE_FILES})

target_link_libraries(${PHWC4_TO_BHWC_TARGET_NAME}
    common_shape
    gl_command_queue
    gl_buffer
    gl_program
    gl_shader
)

install(
    TARGETS ${PHWC4_TO_BHWC_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)
