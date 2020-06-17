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
    nnapi_delegate
    evaluation_utils
    time
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
    absl::strings
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

######################### inference_profiler_stage ###########################

set(INFERENCE_PROFILER_STAGE_TARGET_NAME inference_profiler_stage)

set(INFERENCE_PROFILER_SOURCE_FILES
    inference_profiler_stage.cc
)

add_library(${INFERENCE_PROFILER_STAGE_TARGET_NAME} SHARED
  ${INFERENCE_PROFILER_SOURCE_FILES}
)

target_link_libraries(${INFERENCE_PROFILER_STAGE_TARGET_NAME}
    evaluation_proto
    tf_logging
    tflite_inference_stage
    image_preprocessing_stage
    topk_accuracy_eval_stage
)

install(
    TARGETS ${INFERENCE_PROFILER_STAGE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

######################### object_detection_average_precision_stage ###########################

set(OBJECT_DETECTION_AVERAGE_PRECISION_STAGE_TARGET_NAME object_detection_average_precision_stage)

set(OBJECT_DETECTION_AVERAGE_PRECISION_SOURCE_FILES
    object_detection_average_precision_stage.cc
)

add_library(${OBJECT_DETECTION_AVERAGE_PRECISION_STAGE_TARGET_NAME} SHARED
  ${OBJECT_DETECTION_AVERAGE_PRECISION_SOURCE_FILES}
)

target_link_libraries(${OBJECT_DETECTION_AVERAGE_PRECISION_STAGE_TARGET_NAME}
    evaluation_proto
    tf_logging
    image_metrics
)

install(
    TARGETS ${OBJECT_DETECTION_AVERAGE_PRECISION_STAGE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

######################### object_detection_stage ###########################

set(OBJECT_DETECTION_STAGE_TARGET_NAME object_detection_stage)

set(OBJECT_DETECTION_STAGE_SOURCE_FILES
    object_detection_stage.cc
)

add_library(${OBJECT_DETECTION_STAGE_TARGET_NAME} SHARED
  ${OBJECT_DETECTION_STAGE_TARGET_NAME}
)

target_link_libraries(${OBJECT_DETECTION_STAGE_TARGET_NAME}
    image_preprocessing_stage
    evaluation_proto
    tf_logging
    tflite_inference_stage
    object_detection_average_precision_stage
)

install(
    TARGETS ${OBJECT_DETECTION_STAGE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

add_subdirectory(utils)
