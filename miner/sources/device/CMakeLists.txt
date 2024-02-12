
set(DEVICE_HEADERS
amd.hpp
device_manager.hpp
nvidia.hpp
)

set(DEVICE_SOURCES
amd.cpp
device.cpp
device_manager.cpp
nvidia.cpp
)

target_sources(${MINER_EXE} PUBLIC
${DEVICE_HEADERS}
${DEVICE_SOURCES}
)
