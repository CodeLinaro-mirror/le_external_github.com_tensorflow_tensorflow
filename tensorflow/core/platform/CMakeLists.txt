########################### env_time #############################

set(ENV_TIME_TARGET_NAME env_time)

set(ENV_TIME_SOURCE_FILES
  default/env_time.cc
)

add_library(${ENV_TIME_TARGET_NAME} SHARED ${ENV_TIME_SOURCE_FILES})

install(
    TARGETS ${ENV_TIME_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### tf_logging #############################

set(TF_LOGGING_TARGET_NAME tf_logging)

set(TF_LOGGING_SOURCE_FILES
  default/logging.cc
)

add_library(${TF_LOGGING_TARGET_NAME} SHARED ${TF_LOGGING_SOURCE_FILES})

target_link_libraries(${TF_LOGGING_TARGET_NAME}
  absl
  ${ENV_TIME_TARGET_NAME}
)

install(
    TARGETS ${TF_LOGGING_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)
