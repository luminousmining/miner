
set(NETWORK_HEADERS
network.hpp
)

set(NETWORK_SOURCES
network.cpp
network_callback.cpp
)

target_sources(${MINER_EXE} PUBLIC
${NETWORK_HEADERS}
${NETWORK_SOURCES}
)
