
set(COMMON_LOG_HEADERS
log_type.hpp
log.hpp
log_display.hpp
)

set(COMMON_LOG_SOURCES
log.cpp
log_display.cpp
)

target_sources(${MINER_EXE} PUBLIC
${COMMON_LOG_HEADERS}
${COMMON_LOG_SOURCES}
)
