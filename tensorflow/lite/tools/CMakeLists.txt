########################### command_line_flags ################################
set(COMMAND_LINE_FLAGS_TARGET_NAME command_line_flags)

set(COMMAND_LINE_FLAGS_SOURCE_FILES
    command_line_flags.cc
)

add_library(${COMMAND_LINE_FLAGS_TARGET_NAME} SHARED ${COMMAND_LINE_FLAGS_SOURCE_FILES})

target_link_libraries(${COMMAND_LINE_FLAGS_TARGET_NAME} ${TFLITE_TARGET_NAME}
  minimal_logging
)

install(
    TARGETS ${COMMAND_LINE_FLAGS_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

add_subdirectory(benchmark)
add_subdirectory(evaluation)
add_subdirectory(accuracy/ilsvrc)
