set(PROGPOW_HEADERS
firopow.hpp
kawpow.hpp
progpow.hpp
result.hpp
cuda/progpow.cuh
)

set(PROGPOW_SOURCES
progpow.cpp
progpow_write_kernel_cuda.cpp
progpow_write_kernel_opencl.cpp
)

target_sources(${MINER_EXE} PUBLIC
${PROGPOW_HEADERS}
${PROGPOW_SOURCES}
)
