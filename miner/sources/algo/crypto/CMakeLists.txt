set(CRYPTO_HEADERS
fnv1.hpp
kiss99.hpp
)

set(CRYPTO_SOURCES
fnv1.cpp
kiss99.cpp
)

target_sources(${MINER_EXE} PUBLIC
${CRYPTO_HEADERS}
${CRYPTO_SOURCES}
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${CRYPTO_HEADERS}
${CRYPTO_SOURCES}
)

add_subdirectory(cuda)
add_subdirectory(opencl)
add_subdirectory(tests)
