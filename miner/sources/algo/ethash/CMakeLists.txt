set(ETHASH_HEADERS
ethash.hpp
result.hpp
cuda/ethash.cuh
)

set(ETHASH_SOURCES
ethash.cpp
)

target_sources(${MINER_EXE} PUBLIC
${ETHASH_HEADERS}
${ETHASH_SOURCES}
)

add_subdirectory(opencl)
