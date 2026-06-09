file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")

if (BUILD_EXE_MINER)
    # blake3_pow.cpp is the host PoW oracle used only by the unit tests; it (and the vendored
    # blake3_ref it links) are not needed for a mining session, so only the headers ship here.
    target_sources(${MINER_EXE} PUBLIC
        ${HEADERS}
    )
endif()

if (BUILD_EXE_UNIT_TEST)
    target_sources(${UNIT_TEST_EXE} PUBLIC
        ${HEADERS}
        ${SOURCES}
    )
    target_link_libraries(${UNIT_TEST_EXE} PUBLIC blake3_ref)
endif()

add_subdirectory(opencl)
add_subdirectory(tests)

set(SOURCES_ALGO_BLAKE3 ${HEADERS} ${SOURCES} PARENT_SCOPE)
