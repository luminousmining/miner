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

set(OUT_PROGPOW ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/progpow)

file(COPY result.hpp DESTINATION ${OUT_PROGPOW})

add_subdirectory(cuda)
add_subdirectory(opencl)

set(SOURCES_ALGO_PROGPOW ${HEADERS} ${SOURCES} PARENT_SCOPE)
