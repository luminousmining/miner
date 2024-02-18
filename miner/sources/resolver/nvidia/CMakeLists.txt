set(RESOLVER_NVIDIA_HEADERS
nvidia.hpp
autolykos_v2.hpp
autolykos_v2_kernel_parameter.hpp
ethash.hpp
etchash.hpp
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
