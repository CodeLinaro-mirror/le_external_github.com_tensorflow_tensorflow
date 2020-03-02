include_directories(util)

########################### stats_calculator_portable #############################
set(STATS_CALCULATOR_PORTABLE_TARGET_NAME stats_calculator_portable)

set(STATS_CALCULATOR_PORTABLE_SOURCE_FILES
    util/stats_calculator.cc
)

add_library(${STATS_CALCULATOR_PORTABLE_TARGET_NAME} SHARED ${STATS_CALCULATOR_PORTABLE_SOURCE_FILES})

install(
    TARGETS ${STATS_CALCULATOR_PORTABLE_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)
