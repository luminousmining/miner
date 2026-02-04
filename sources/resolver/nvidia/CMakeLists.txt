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

add_subdirectory(tests)

set(SOURCES_RESOLVER_NVIDIA ${HEADERS} ${SOURCES} PARENT_SCOPE)
set(SOURCES_RESOLVER_NVIDIA_TESTS ${SOURCES_RESOLVER_NVIDIA_TESTS} PARENT_SCOPE)
