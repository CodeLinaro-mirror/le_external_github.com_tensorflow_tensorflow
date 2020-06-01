########################### nnapi_delegate #############################
set(NNAPI_DELEGATE_TARGET_NAME nnapi_delegate)

set(NNAPI_DELEGATE_SOURCE_FILES
    nnapi_delegate.cc
    quant_lstm_sup.cc
)

add_library(${NNAPI_DELEGATE_TARGET_NAME} SHARED ${NNAPI_DELEGATE_SOURCE_FILES})

target_link_libraries(${NNAPI_DELEGATE_TARGET_NAME}
    ${TFLITE_TARGET_NAME}
    rt
)

install(
    TARGETS ${NNAPI_DELEGATE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)
