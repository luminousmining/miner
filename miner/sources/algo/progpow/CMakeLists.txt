set(PROGPOW_HEADERS
firopow.hpp
kawpow.hpp
progpow.hpp
result.hpp
cuda/progpow.cuh
)

set(PROGPOW_SOURCES
progpow.cpp
kernel_cuda.cpp
kernel_opencl.cpp
)

target_sources(${MINER_EXE} PUBLIC
${PROGPOW_HEADERS}
${PROGPOW_SOURCES}
)


set(OUT_PROGPOW ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/progpow)

file(COPY result.hpp DESTINATION ${OUT_PROGPOW})

add_subdirectory(cuda)
add_subdirectory(opencl)
