add_subdirectory(proto)
add_subdirectory(stages)
add_subdirectory(tasks/coco_object_detection)
add_subdirectory(tasks/imagenet_image_classification)
add_subdirectory(tasks/inference_diff)

########################### utils #############################
set(UTILS_TARGET_NAME evaluation_utils)

set(UTILS_SOURCE_FILES
    utils.cc
)

add_library(${UTILS_TARGET_NAME} SHARED ${UTILS_SOURCE_FILES})

target_link_libraries(${UTILS_TARGET_NAME}
    ${TFLITE_TARGET_NAME}
    xnnpack_delegate
    nnapi_delegate
    gpu_delegate
    gl_delegate
)

install(
    TARGETS ${UTILS_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)
