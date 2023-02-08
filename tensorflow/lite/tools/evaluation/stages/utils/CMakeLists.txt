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

if (TFLITE_C_BUILD_SHARED_LIBS)
    target_link_libraries(${IMAGE_METRICS_TARGET_NAME}
        tensorflowlite_c
    )
else ()
    target_link_libraries(${IMAGE_METRICS_TARGET_NAME}
        tensorflow-lite
    )
endif()

install(TARGETS ${IMAGE_METRICS_TARGET_NAME}
    LIBRARY
)
