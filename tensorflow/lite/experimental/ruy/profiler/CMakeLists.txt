########################### ruy_profiler ################################
set(RUY_PROFILER_TARGET_NAME ruy_profiler)

set(RUY_PROFILER_SOURCE_FILES
    profiler.cc
    treeview.cc
)

add_library(${RUY_PROFILER_TARGET_NAME} SHARED ${RUY_PROFILER_SOURCE_FILES})


install(
    TARGETS ${RUY_PROFILER_TARGET_NAME}
    LIBRARY DESTINATION ${TFLITE_INSTALL_LIBDIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)
