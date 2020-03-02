find_package(Protobuf REQUIRED)

########################### evaluation_process_proto ###########################

set(EVALUATION_PROCESS_PROTO_TARGET evaluation_process_proto)

set(EVALUATION_PROCESS_PROTO_SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/evaluation_stages.proto
  ${CMAKE_CURRENT_SOURCE_DIR}/evaluation_config.proto
)
set(EVALUATION_PROCESS_PROTO_OUTPUT
  evaluation_stages.pb.cc
  evaluation_config.pb.cc
)

add_custom_target(
  ${EVALUATION_PROCESS_PROTO_TARGET} ALL
  DEPENDS ${EVALUATION_PROCESS_PROTO_OUTPUT}
)

add_custom_command(
  OUTPUT ${EVALUATION_PROCESS_PROTO_OUTPUT}
  COMMAND protobuf::protoc
      --cpp_out=${TFLITE_TOP_BINARY_DIR}
      --proto_path=${TFLITE_TOP_SOURCE_DIR}
      ${EVALUATION_PROCESS_PROTO_SOURCE}
  DEPENDS ${EVALUATION_PROCESS_PROTO_SOURCE} protobuf::protoc
  COMMENT ""
)

############################## evaluation_proto ################################

set(EVALUATION_PROTO_TARGET evaluation_proto)

add_library(${EVALUATION_PROTO_TARGET} SHARED
  ${EVALUATION_PROCESS_PROTO_OUTPUT}
)

target_include_directories(${EVALUATION_PROTO_TARGET} PUBLIC
  ${TFLITE_TOP_BINARY_DIR}
)

target_link_libraries(${EVALUATION_PROTO_TARGET} ${Protobuf_LIBRARIES})

install(
    TARGETS ${EVALUATION_PROTO_TARGET}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)
