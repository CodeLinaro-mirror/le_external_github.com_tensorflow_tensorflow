######################### tflite_inference_stage ###########################

set(TFLITE_INFERENCE_STAGE_TARGET_NAME tflite_inference_stage)

set(IMAGE_INFERENCE_STAGE_SOURCE_FILES
    tflite_inference_stage.cc
)

add_library(${TFLITE_INFERENCE_STAGE_TARGET_NAME} SHARED
    ${IMAGE_INFERENCE_STAGE_SOURCE_FILES}
)

target_link_libraries(${TFLITE_INFERENCE_STAGE_TARGET_NAME}
    evaluation_proto
    tf_logging
    tensorflow-lite
)

install(
    TARGETS ${TFLITE_INFERENCE_STAGE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

######################### image_preprocessing_stage ###########################

set(IMAGE_PREPROCESSING_STAGE_TARGET_NAME image_preprocessing_stage)

set(IMAGE_PREPROCESSING_STAGE_SOURCE_FILES
    image_preprocessing_stage.cc
)

add_library(${IMAGE_PREPROCESSING_STAGE_TARGET_NAME} SHARED
    ${IMAGE_PREPROCESSING_STAGE_SOURCE_FILES}
)

target_link_libraries(${IMAGE_PREPROCESSING_STAGE_TARGET_NAME}
    evaluation_proto
    tf_logging
    time
    jpeg_internal
)

install(
    TARGETS ${IMAGE_PREPROCESSING_STAGE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

######################### topk_accuracy_eval_stage ###########################

set(TOPK_ACCURACY_EVAL_STAGE_TARGET_NAME topk_accuracy_eval_stage)

set(TOPK_ACCURACY_EVAL_SOURCE_FILES
    topk_accuracy_eval_stage.cc
)

add_library(${TOPK_ACCURACY_EVAL_STAGE_TARGET_NAME} SHARED
    ${TOPK_ACCURACY_EVAL_SOURCE_FILES}
)

target_link_libraries(${TOPK_ACCURACY_EVAL_STAGE_TARGET_NAME}
    evaluation_proto
    tf_logging
)

install(
    TARGETS ${TOPK_ACCURACY_EVAL_STAGE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

######################### image_classification_stage ###########################

set(IMAGE_CLASSIFICATION_STAGE_TARGET_NAME image_classification_stage)

set(IMAGE_CLASSIFICATION_STAGE_SOURCE_FILES
    image_classification_stage.cc
)

add_library(${IMAGE_CLASSIFICATION_STAGE_TARGET_NAME} SHARED
  ${IMAGE_CLASSIFICATION_STAGE_SOURCE_FILES}
)

target_link_libraries(${IMAGE_CLASSIFICATION_STAGE_TARGET_NAME}
    evaluation_proto
    tf_logging
    tflite_inference_stage
    image_preprocessing_stage
    topk_accuracy_eval_stage
)

install(
    TARGETS ${IMAGE_CLASSIFICATION_STAGE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)
