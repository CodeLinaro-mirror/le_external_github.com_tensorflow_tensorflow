set(JPEG_INTERNAL_TARGET_NAME jpeg_internal)

set(JPEG_INTERNAL_SOURCE_FILES
    lib/jpeg/jpeg_handle.cc
    lib/jpeg/jpeg_mem.cc
)

add_library(${JPEG_INTERNAL_TARGET_NAME} SHARED
    ${JPEG_INTERNAL_SOURCE_FILES}
)

target_link_libraries(${JPEG_INTERNAL_TARGET_NAME}
    absl::strings
    Eigen3::Eigen
    libjpeg.a
    tf_logging
)

install(TARGETS ${JPEG_INTERNAL_TARGET_NAME}
    LIBRARY
)
