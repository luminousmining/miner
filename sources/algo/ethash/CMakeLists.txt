file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")

if (BUILD_EXE_MINER)
    target_sources(${MINER_EXE} PUBLIC
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

add_subdirectory(opencl)
add_subdirectory(tests)

set(SOURCES_ALGO_ETHASH ${HEADERS} ${SOURCES} PARENT_SCOPE)
set(SOURCES_ALGO_ETHASH_TESTS ${SOURCES_ALGO_ETHASH_TESTS} PARENT_SCOPE)
