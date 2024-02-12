
set(COMMON_HEADERS
boost_utils.hpp
cast.hpp
chrono.hpp
cli.hpp
config.hpp
custom.hpp
dashboard.hpp
date.hpp
kernel_generator.hpp
system.hpp
)

set(COMMON_SOURCES
chrono.cpp
cli.cpp
cli_device_custom.cpp
cli_device_setting.cpp
cli_pool.cpp
config.cpp
dashboard.cpp
kernel_generator.cpp
)

target_sources(${MINER_EXE} PUBLIC
${COMMON_HEADERS}
${COMMON_SOURCES}
)

add_subdirectory(error)
add_subdirectory(log)
