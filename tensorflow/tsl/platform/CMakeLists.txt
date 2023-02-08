set(ENV_TIME_TARGET_NAME env_time)

set(ENV_TIME_SOURCE_FILES
  default/env_time.cc
)

add_library(${ENV_TIME_TARGET_NAME} SHARED ${ENV_TIME_SOURCE_FILES})

target_include_directories(${ENV_TIME_TARGET_NAME}
  PUBLIC
    ${TFLITE_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${ML_DTYPES_SOURCE_DIR}/ml_dtypes>
)

set(ENV_TIME_DEPENDENCIES
  absl::strings
  Eigen3::Eigen
)

if(CMAKE_SYSTEM_NAME MATCHES "Android")
  list(APPEND ENV_TIME_DEPENDENCIES
    ${ANDROID_LOG_LIB}
  )
endif()

target_link_libraries(${ENV_TIME_TARGET_NAME}
  ${ENV_TIME_DEPENDENCIES}
)

install(TARGETS ${ENV_TIME_TARGET_NAME}
  LIBRARY
)

set(TF_LOGGING_TARGET_NAME tf_logging)

set(TF_LOGGING_SOURCE_FILES
  default/logging.cc
  default/mutex.cc
)

add_library(${TF_LOGGING_TARGET_NAME} SHARED ${TF_LOGGING_SOURCE_FILES})

target_link_libraries(${TF_LOGGING_TARGET_NAME}
  absl::base
  nsync_cpp
  ml_dtypes
  ${ENV_TIME_TARGET_NAME}
)

install(TARGETS ${TF_LOGGING_TARGET_NAME}
  LIBRARY
)
