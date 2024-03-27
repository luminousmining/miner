
set(COMMON_LOG_HEADERS
log.hpp
log_display.hpp
log_file.hpp
log_type.hpp
)

set(COMMON_LOG_SOURCES
log.cpp
log_display.cpp
log_file.cpp
)

target_sources(${MINER_EXE} PUBLIC
${COMMON_LOG_HEADERS}
${COMMON_LOG_SOURCES}
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${COMMON_LOG_HEADERS}
${COMMON_LOG_SOURCES}
)
