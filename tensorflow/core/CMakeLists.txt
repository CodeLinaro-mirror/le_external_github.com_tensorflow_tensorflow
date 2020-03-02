########################### jpeg_internal #############################

set(JPEG_INTERNAL_TARGET_NAME jpeg_internal)

set(JPEG_INTERNAL_SOURCE_FILES
    lib/jpeg/jpeg_handle.cc
    lib/jpeg/jpeg_mem.cc
)

add_library(${JPEG_INTERNAL_TARGET_NAME} SHARED
    ${JPEG_INTERNAL_SOURCE_FILES}
)

target_link_libraries(${JPEG_INTERNAL_TARGET_NAME}
    absl
    jpeg
    tf_logging
)

install(
    TARGETS ${JPEG_INTERNAL_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### stats_calculator_portable #############################
set(STATS_CALCULATOR_PORTABLE_TARGET_NAME stats_calculator_portable)

set(STATS_CALCULATOR_PORTABLE_SOURCE_FILES
    util/stats_calculator.cc
)

add_library(${STATS_CALCULATOR_PORTABLE_TARGET_NAME} SHARED
    ${STATS_CALCULATOR_PORTABLE_SOURCE_FILES}
)

target_include_directories(${STATS_CALCULATOR_PORTABLE_TARGET_NAME} PRIVATE
    util
)

install(
    TARGETS ${STATS_CALCULATOR_PORTABLE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    GROUP_EXECUTE GROUP_READ
    WORLD_EXECUTE WORLD_READ
    )

add_subdirectory(platform)