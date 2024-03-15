
set(COMMON_ERROR_HEADERS
cuda_error.hpp
opencl_error.hpp
)

set(COMMON_ERROR_SOURCES
opencl_error.cpp
)

target_sources(${MINER_EXE} PUBLIC
${COMMON_ERROR_HEADERS}
${COMMON_ERROR_SOURCES}
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${COMMON_ERROR_HEADERS}
${COMMON_ERROR_SOURCES}
)