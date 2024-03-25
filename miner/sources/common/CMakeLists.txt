set(COMMON_HEADERS
boost_utils.hpp
cast.hpp
chrono.hpp
cli.hpp
config.hpp
custom.hpp
dashboard.hpp
date.hpp
formater_hashrate.hpp
kernel_generator.hpp
system.hpp
)

set(COMMON_SOURCES
boost_utils.cpp
chrono.cpp
cli.cpp
cli_dev.cpp
cli_device_custom.cpp
cli_device_setting.cpp
cli_pool.cpp
cli_smart_mining.cpp
config.cpp
dashboard.cpp
kernel_generator.cpp
)

target_sources(${MINER_EXE} PUBLIC
${COMMON_HEADERS}
${COMMON_SOURCES}
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${COMMON_HEADERS}
${COMMON_SOURCES}
)

add_subdirectory(cuda)
add_subdirectory(error)
add_subdirectory(log)
add_subdirectory(opencl)
add_subdirectory(mocker)
