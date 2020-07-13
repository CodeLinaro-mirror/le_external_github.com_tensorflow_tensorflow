########################### time #############################
set(TIME_TARGET_NAME time)

set(TIME_SOURCE_FILES
    time.cc
)

add_library(${TIME_TARGET_NAME} SHARED ${TIME_SOURCE_FILES})

install(
    TARGETS ${TIME_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################## memory_info #############################
set(MEMORY_INFO_TARGET_NAME memory_info)

set(MEMORY_INFO_SOURCE_FILES
    memory_info.cc
)

add_library(${MEMORY_INFO_TARGET_NAME} SHARED ${MEMORY_INFO_SOURCE_FILES})

install(
    TARGETS ${MEMORY_INFO_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### profile_summarizer #############################
set (PROFILE_SUMMARIZER_TARGET_NAME profile_summarizer)

set(PROFILE_SUMMARIZER_SOURCE_FILES
    profile_summarizer.cc
)

add_library(${PROFILE_SUMMARIZER_TARGET_NAME} SHARED ${PROFILE_SUMMARIZER_SOURCE_FILES})

target_link_libraries(${PROFILE_SUMMARIZER_TARGET_NAME}
    stats_calculator_portable
)

install(
    TARGETS ${PROFILE_SUMMARIZER_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### profile_summarizer_formatter #############################
set (PROFILE_SUMMARIZER_FORMATTER_TARGET_NAME profile_summarizer_formatter)

set(PROFILE_SUMMARIZER_FORMATTER_SOURCE_FILES
    profile_summary_formatter.cc
)

add_library(${PROFILE_SUMMARIZER_FORMATTER_TARGET_NAME} SHARED ${PROFILE_SUMMARIZER_FORMATTER_SOURCE_FILES})

target_link_libraries(${PROFILE_SUMMARIZER_FORMATTER_TARGET_NAME}
    stats_calculator_portable
)

install(
    TARGETS ${PROFILE_SUMMARIZER_FORMATTER_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

