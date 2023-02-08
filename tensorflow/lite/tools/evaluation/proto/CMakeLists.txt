# Compilation of Protobuf files
find_package(Protobuf REQUIRED)

########################### evaluation_process_proto ###########################

set(EVALUATION_PROCESS_PROTO_TARGET evaluation_process_proto)

set(EVALUATION_PROCESS_PROTO_SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/evaluation_stages.proto
  ${CMAKE_CURRENT_SOURCE_DIR}/evaluation_config.proto
  ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing_steps.proto
)
set(EVALUATION_PROCESS_PROTO_OUTPUT
  ${CMAKE_CURRENT_SOURCE_DIR}/evaluation_stages.pb.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/evaluation_config.pb.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing_steps.pb.cc
)

add_custom_target(
  ${EVALUATION_PROCESS_PROTO_TARGET} ALL
  DEPENDS ${EVALUATION_PROCESS_PROTO_OUTPUT}
)

add_custom_command(
  OUTPUT ${EVALUATION_PROCESS_PROTO_OUTPUT}
  COMMAND protobuf::protoc
      --cpp_out=${TENSORFLOW_SOURCE_DIR}
      --proto_path=${TENSORFLOW_SOURCE_DIR}
      ${EVALUATION_PROCESS_PROTO_SOURCE}
  DEPENDS ${EVALUATION_PROCESS_PROTO_SOURCE} protobuf::protoc
  COMMENT ""
)

############################## evaluation_proto ################################

set(EVALUATION_PROTO_TARGET evaluation_proto)

add_library(${EVALUATION_PROTO_TARGET} SHARED
  ${EVALUATION_PROCESS_PROTO_OUTPUT}
)

set(EVALUATION_PROTO_DEPENDENCIES libprotobuf.a)

if(CMAKE_SYSTEM_NAME MATCHES "Android")
  list(APPEND EVALUATION_PROTO_DEPENDENCIES
    ${ANDROID_LOG_LIB}
  )
endif()

target_link_libraries(${EVALUATION_PROTO_TARGET}
  ${EVALUATION_PROTO_DEPENDENCIES}
)

install(TARGETS ${EVALUATION_PROTO_TARGET}
  LIBRARY
)
