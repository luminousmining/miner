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

add_subdirectory(cuda)
add_subdirectory(opencl)
add_subdirectory(tests)

set(SOURCES_ALGO_CRYPTO ${HEADERS} ${SOURCES} PARENT_SCOPE)
set(SOURCES_ALGO_CRYPTO_TESTS ${SOURCES_ALGO_CRYPTO_TESTS} PARENT_SCOPE)
