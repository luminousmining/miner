set(PROGPOW_HEADERS
evrprogpow.hpp
firopow.hpp
kawpow.hpp
progpow.hpp
result.hpp
cuda/progpow.cuh
)

set(PROGPOW_SOURCES
evrprogpow.cpp
firopow.cpp
kawpow.cpp
kernel_cuda.cpp
kernel_opencl.cpp
progpow.cpp
)

target_sources(${MINER_EXE} PUBLIC
${PROGPOW_HEADERS}
${PROGPOW_SOURCES}
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${PROGPOW_HEADERS}
${PROGPOW_SOURCES}
)


set(OUT_PROGPOW ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/progpow)

file(COPY result.hpp DESTINATION ${OUT_PROGPOW})

add_subdirectory(cuda)
add_subdirectory(opencl)
