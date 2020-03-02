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