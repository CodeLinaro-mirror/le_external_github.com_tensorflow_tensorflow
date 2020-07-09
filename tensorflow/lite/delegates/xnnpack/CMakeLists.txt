
########################### xnnpack_delegate #############################
set(XNNPACK_DELEGATE_TARGET_NAME xnnpack_delegate)

set(XNNPACK_DELEGATE_SOURCE_FILES
    xnnpack_delegate.cc
)

add_library(${XNNPACK_DELEGATE_TARGET_NAME} SHARED ${XNNPACK_DELEGATE_SOURCE_FILES})

target_compile_definitions(${XNNPACK_DELEGATE_TARGET_NAME} PRIVATE
    ASSERTIONS=2
    ERROR_ON_UNDEFINED_SYMBOLS=1
    DEMANGLE_SUPPORT=1
    EXIT_RUNTIME=1
    ALLOW_MEMORY_GROWTH=1
    TOTAL_MEMORY=134217728
)

target_link_libraries(${XNNPACK_DELEGATE_TARGET_NAME} ${TFLITE_TARGET_NAME}
    XNNPACK
)

install(
    TARGETS ${XNNPACK_DELEGATE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### conv_2d_test #############################
set(CONV_2D_TEST_TARGET_NAME conv_2d_test)

set(CONV_2D_TEST_TARGET_SOURCE_FILES
    conv_2d_test.cc
)

add_executable(${CONV_2D_TEST_TARGET_NAME} ${CONV_2D_TEST_TARGET_SOURCE_FILES})

target_link_libraries(${CONV_2D_TEST_TARGET_NAME}
    xnnpack_delegate
    gtest
    gtest_main
)

install(
    TARGETS ${CONV_2D_TEST_TARGET_NAME}
    RUNTIME DESTINATION ${TFLITE_INSTALL_BINDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

########################### depthwise_conv_2d_test #############################
set(DEPTHWISE_CONV_2D_TEST_TARGET_NAME depthwise_conv_2d_test)

set(DEPTHWISE_CONV_2D_TEST_TARGET_SOURCE_FILES
    depthwise_conv_2d_test.cc
)

add_executable(${DEPTHWISE_CONV_2D_TEST_TARGET_NAME} ${DEPTHWISE_CONV_2D_TEST_TARGET_SOURCE_FILES})

target_link_libraries(${DEPTHWISE_CONV_2D_TEST_TARGET_NAME}
    xnnpack_delegate
    gtest
    gtest_main
)

install(
    TARGETS ${DEPTHWISE_CONV_2D_TEST_TARGET_NAME}
    RUNTIME DESTINATION ${TFLITE_INSTALL_BINDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)