file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")

target_sources(${MINER_EXE} PUBLIC
${HEADERS}
${SOURCES}
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${HEADERS}
${SOURCES}
)

add_subdirectory(opencl)

set(SOURCES_ALGO_AUTOLYOS ${HEADERS} ${SOURCES} PARENT_SCOPE)
