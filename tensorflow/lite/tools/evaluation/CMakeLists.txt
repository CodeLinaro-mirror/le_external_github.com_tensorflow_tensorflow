add_subdirectory(proto)
add_subdirectory(stages)

########################### utils #############################
set(UTILS_TARGET_NAME evaluation_utils)

set(UTILS_SOURCE_FILES
    utils.cc
)

add_library(${UTILS_TARGET_NAME} SHARED ${UTILS_SOURCE_FILES})

target_link_libraries(${UTILS_TARGET_NAME} ${TFLITE_TARGET_NAME})

install(
    TARGETS ${UTILS_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)
