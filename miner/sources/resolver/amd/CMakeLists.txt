set(RESOLVER_AMD_HEADERS
amd.hpp
autolykos_v2.hpp
autolykos_v2_kernel_parameter.hpp
etchash.hpp
ethash.hpp
ethash_kernel_parameter.hpp
evrprogpow.hpp
firopow.hpp
kawpow.hpp
progpow.hpp
progpow_kernel_parameter.hpp
)

set(RESOLVER_AMD_SOURCES
amd.cpp
autolykos_v2.cpp
etchash.cpp
ethash.cpp
evrprogpow.cpp
firopow.cpp
kawpow.cpp
progpow.cpp
)

target_sources(${MINER_EXE} PUBLIC
${RESOLVER_AMD_HEADERS}
${RESOLVER_AMD_SOURCES}
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${RESOLVER_AMD_HEADERS}
${RESOLVER_AMD_SOURCES}
)

add_subdirectory(tests)
