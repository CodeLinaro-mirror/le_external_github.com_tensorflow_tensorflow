########################### imagenet_model_evaluator #############################

set(IMAGENET_MODEL_EVALUATOR_TARGET_NAME imagenet_model_evaluator)

set(IMAGENET_MODEL_EVALUATOR_SOURCE_FILES
    imagenet_model_evaluator.cc
)

add_library(${IMAGENET_MODEL_EVALUATOR_TARGET_NAME} SHARED
    ${IMAGENET_MODEL_EVALUATOR_SOURCE_FILES}
)

target_link_libraries(${IMAGENET_MODEL_EVALUATOR_TARGET_NAME}
    evaluation_proto
    command_line_flags
    tf_logging
    image_classification_stage
)

install(
    TARGETS ${IMAGENET_MODEL_EVALUATOR_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### imagenet_accuracy_eval #############################

set(IMAGENET_ACCURACY_EVAL_TARGET_NAME imagenet_accuracy_eval)

set(IMAGENET_ACCURACY_EVAL_SOURCE_FILES
    imagenet_accuracy_eval.cc
)

add_executable(${IMAGENET_ACCURACY_EVAL_TARGET_NAME}
    ${IMAGENET_ACCURACY_EVAL_SOURCE_FILES}
)

target_link_libraries(${IMAGENET_ACCURACY_EVAL_TARGET_NAME}
    imagenet_model_evaluator
)

install(
    TARGETS ${IMAGENET_ACCURACY_EVAL_TARGET_NAME}
    RUNTIME DESTINATION ${TFLITE_INSTALL_BINDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)
