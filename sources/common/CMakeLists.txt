file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")

if (BUILD_EXE_MINER)
    target_sources(${MINER_EXE} PUBLIC
        ${HEADERS}
        ${SOURCES}
    )
endif()

if (BUILD_EXE_BENCHMARK)
    target_sources(${BENCH_EXE} PUBLIC
        ${HEADERS}
        ${SOURCES}
    )
endif()

if (BUILD_EXE_UNIT_TEST)
    target_sources(${UNIT_TEST_EXE} PUBLIC
        ${HEADERS}
        ${SOURCES}
    )
endif()

add_subdirectory(cli)
add_subdirectory(cuda)
add_subdirectory(error)
add_subdirectory(kernel_generator)
add_subdirectory(log)
add_subdirectory(opencl)
add_subdirectory(mocker)
add_subdirectory(tool)


set(SOURCES_COMMON ${HEADERS} ${SOURCES} PARENT_SCOPE)
set(SOURCES_COMMON_CLI ${SOURCES_COMMON_CLI} PARENT_SCOPE)
set(SOURCES_COMMON_ERROR ${SOURCES_COMMON_ERROR} PARENT_SCOPE)
set(SOURCES_COMMON_KERNEL_GENERATOR ${SOURCES_COMMON_KERNEL_GENERATOR} PARENT_SCORE)
set(SOURCES_COMMON_LOG ${SOURCES_COMMON_LOG} PARENT_SCOPE)
set(SOURCES_COMMON_MOCKER ${SOURCES_COMMON_MOCKER} PARENT_SCOPE)
set(SOURCES_COMMON_TOOL ${SOURCES_COMMON_TOOL} PARENT_SCOPE)
