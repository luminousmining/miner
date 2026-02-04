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

set(SOURCES_PROFILER ${HEADERS} ${SOURCES} PARENT_SCOPE)
set(SOURCES_PROFILER_ADL ${SOURCES_PROFILER_ADL} PARENT_SCOPE)
