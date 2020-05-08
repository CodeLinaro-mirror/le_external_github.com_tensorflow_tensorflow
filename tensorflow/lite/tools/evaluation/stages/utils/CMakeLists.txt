######################### image_metrics ###########################

set(IMAGE_METRICS_TARGET_NAME image_metrics)

set(IMAGE_METRICS_SOURCE_FILES
    image_metrics.cc
)

add_library(${IMAGE_METRICS_TARGET_NAME} SHARED
    ${IMAGE_METRICS_SOURCE_FILES}
)

target_link_libraries(${IMAGE_METRICS_TARGET_NAME}
    tf_logging
    absl::flat_hash_map
)

install(
    TARGETS ${IMAGE_METRICS_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)