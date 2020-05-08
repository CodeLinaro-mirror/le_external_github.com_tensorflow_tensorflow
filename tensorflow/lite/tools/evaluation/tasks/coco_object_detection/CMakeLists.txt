########################## run_eval ################################
set(RUN_EVAL_TARGET_NAME run_eval_detection)

set(RUN_EVAL_SOURCE_FILES
    run_eval.cc
)

add_executable(${RUN_EVAL_TARGET_NAME} ${RUN_EVAL_SOURCE_FILES})

target_link_libraries(${RUN_EVAL_TARGET_NAME}
   ${TFLITE_TARGET_NAME}
    tf_logging
    command_line_flags
    evaluation_utils
    evaluation_proto
    object_detection_stage
    absl::flat_hash_map
)

install(
    TARGETS ${RUN_EVAL_TARGET_NAME}
    RUNTIME DESTINATION ${TFLITE_INSTALL_BINDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)
