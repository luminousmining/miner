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

set(SOURCES_STRATISTICAL ${HEADERS} ${SOURCES} PARENT_SCOPE)
