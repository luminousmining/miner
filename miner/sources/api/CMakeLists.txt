set(API_HEADERS
api.hpp
)

set(API_SOURCES
api.cpp
)

target_sources(${MINER_EXE} PUBLIC
${API_HEADERS}
${API_SOURCES}
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${API_HEADERS}
${API_SOURCES}
)
