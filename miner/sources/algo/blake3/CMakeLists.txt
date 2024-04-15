set(BLAKE3_HEADERS
result.hpp
)

set(BLAKE3_SOURCES
)

target_sources(${MINER_EXE} PUBLIC
${BLAKE3_HEADERS}
${BLAKE3_SOURCES}
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${BLAKE3_HEADERS}
${BLAKE3_SOURCES}
)

add_subdirectory(opencl)
