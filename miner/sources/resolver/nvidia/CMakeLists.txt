set(RESOLVER_NVIDIA_HEADERS
nvidia.hpp
autolykos_v2.hpp
autolykos_v2_kernel_parameter.hpp
blake3.hpp
blake3_kernel_parameter.hpp
etchash.hpp
ethash.hpp
ethash_kernel_parameter.hpp
evrprogpow.hpp
firopow.hpp
kawpow.hpp
progpow.hpp
progpow_kernel_parameter.hpp
)

set(RESOLVER_NVIDIA_SOURCES
nvidia.cpp
autolykos_v2.cpp
blake3.cpp
etchash.cpp
ethash.cpp
evrprogpow.cpp
firopow.cpp
kawpow.cpp
progpow.cpp
)

target_sources(${MINER_EXE} PUBLIC
${RESOLVER_NVIDIA_HEADERS}
${RESOLVER_NVIDIA_SOURCES}
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${RESOLVER_NVIDIA_HEADERS}
${RESOLVER_NVIDIA_SOURCES}
)

add_subdirectory(tests)
