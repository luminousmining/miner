set(TEST_RESOLVER_NVIDIA_HEADERS
)

set(TEST_RESOLVER_NVIDIA_SOURCES
autolykos_v2.cpp
blake3.cpp
ethash.cpp
firopow.cpp
kawpow.cpp
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${TEST_RESOLVER_NVIDIA_HEADERS}
${TEST_RESOLVER_NVIDIA_SOURCES}
)
