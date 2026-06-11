file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")

if (BUILD_EXE_MINER)
    target_sources(${MINER_EXE} PUBLIC
        ${HEADERS}
    )
    if (BUILD_CPU)
        # The CPU Blake3 resolver calls algo::blake3::hashRef (blake3_pow.cpp), which
        # links the vendored blake3_ref C lib. Both are otherwise unit-test-only.
        target_sources(${MINER_EXE} PUBLIC
            ${SOURCES}
        )
        target_link_libraries(${MINER_EXE} PUBLIC blake3_ref)
    endif()
endif()

if (BUILD_EXE_UNIT_TEST)
    target_sources(${UNIT_TEST_EXE} PUBLIC
        ${HEADERS}
        ${SOURCES}
    )
    target_link_libraries(${UNIT_TEST_EXE} PUBLIC blake3_ref)
endif()

add_subdirectory(cpu)
add_subdirectory(opencl)
add_subdirectory(tests)

set(SOURCES_ALGO_BLAKE3 ${HEADERS} ${SOURCES} PARENT_SCOPE)
