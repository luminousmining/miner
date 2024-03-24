set(TEST_RESOLVER_AMD_HEADERS
)

set(TEST_RESOLVER_AMD_SOURCES
autolykos_v2.cpp
ethash.cpp
firopow.cpp
kawpow.cpp
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${TEST_RESOLVER_AMD_HEADERS}
${TEST_RESOLVER_AMD_SOURCES}
)
