set(TFLITE_INFERENCE_STAGE_TARGET_NAME tflite_inference_stage)

set(IMAGE_INFERENCE_STAGE_SOURCE_FILES
    tflite_inference_stage.cc
)

add_library(${TFLITE_INFERENCE_STAGE_TARGET_NAME}
    ${IMAGE_INFERENCE_STAGE_SOURCE_FILES}
)

target_link_libraries(${TFLITE_INFERENCE_STAGE_TARGET_NAME}
    evaluation_proto
    tf_logging
)

if (TFLITE_C_BUILD_SHARED_LIBS)
    target_link_libraries(${TFLITE_INFERENCE_STAGE_TARGET_NAME}
        tensorflowlite_c
    )
else ()
    target_link_libraries(${TFLITE_INFERENCE_STAGE_TARGET_NAME}
        tensorflow-lite
    )
endif()

set(IMAGE_PREPROCESSING_STAGE_TARGET_NAME image_preprocessing_stage)

set(IMAGE_PREPROCESSING_STAGE_SOURCE_FILES
    image_preprocessing_stage.cc
)

add_library(${IMAGE_PREPROCESSING_STAGE_TARGET_NAME}
    ${IMAGE_PREPROCESSING_STAGE_SOURCE_FILES}
)

target_link_libraries(${IMAGE_PREPROCESSING_STAGE_TARGET_NAME}
    absl::strings
    evaluation_proto
    tf_logging
    gemmlowp
    ruy
    jpeg_internal
)

set(TOPK_ACCURACY_EVAL_STAGE_TARGET_NAME topk_accuracy_eval_stage)

set(TOPK_ACCURACY_EVAL_SOURCE_FILES
    topk_accuracy_eval_stage.cc
)

add_library(${TOPK_ACCURACY_EVAL_STAGE_TARGET_NAME}
    ${TOPK_ACCURACY_EVAL_SOURCE_FILES}
)

target_link_libraries(${TOPK_ACCURACY_EVAL_STAGE_TARGET_NAME}
    evaluation_proto
    absl::strings
    tf_logging
)

set(INFERENCE_PROFILER_STAGE_TARGET_NAME inference_profiler_stage)

set(INFERENCE_PROFILER_SOURCE_FILES
    inference_profiler_stage.cc
)

add_library(${INFERENCE_PROFILER_STAGE_TARGET_NAME}
    ${INFERENCE_PROFILER_SOURCE_FILES}
)

target_link_libraries(${INFERENCE_PROFILER_STAGE_TARGET_NAME}
    evaluation_proto
    tf_logging
    tflite_inference_stage
    image_preprocessing_stage
    topk_accuracy_eval_stage
)

set(IMAGE_CLASSIFICATION_STAGE_TARGET_NAME image_classification_stage)

set(IMAGE_CLASSIFICATION_STAGE_SOURCE_FILES
    image_classification_stage.cc
)

add_library(${IMAGE_CLASSIFICATION_STAGE_TARGET_NAME}
    ${IMAGE_CLASSIFICATION_STAGE_SOURCE_FILES}
)

target_link_libraries(${IMAGE_CLASSIFICATION_STAGE_TARGET_NAME}
    evaluation_proto
    tf_logging
    tflite_inference_stage
    image_preprocessing_stage
    topk_accuracy_eval_stage
)

set(OBJECT_DETECTION_AVERAGE_PRECISION_STAGE_TARGET_NAME object_detection_average_precision_stage)

set(OBJECT_DETECTION_AVERAGE_PRECISION_SOURCE_FILES
    object_detection_average_precision_stage.cc
)

add_library(${OBJECT_DETECTION_AVERAGE_PRECISION_STAGE_TARGET_NAME}
    ${OBJECT_DETECTION_AVERAGE_PRECISION_SOURCE_FILES}
)

target_link_libraries(${OBJECT_DETECTION_AVERAGE_PRECISION_STAGE_TARGET_NAME}
    evaluation_proto
    tf_logging
    image_metrics
)

set(OBJECT_DETECTION_STAGE_TARGET_NAME object_detection_stage)

set(OBJECT_DETECTION_STAGE_SOURCE_FILES
    object_detection_stage.cc
)

add_library(${OBJECT_DETECTION_STAGE_TARGET_NAME}
    ${OBJECT_DETECTION_STAGE_SOURCE_FILES}
)

target_link_libraries(${OBJECT_DETECTION_STAGE_TARGET_NAME}
    image_preprocessing_stage
    evaluation_proto
    tf_logging
    tflite_inference_stage
    object_detection_average_precision_stage
)

add_subdirectory(utils)
