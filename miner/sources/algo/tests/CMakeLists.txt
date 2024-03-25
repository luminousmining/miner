
set(TEST_ALGO_HEADERS
)

set(TEST_ALGO_SOURCES
algo_type.cpp
math.cpp
rol.cpp
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${TEST_ALGO_HEADERS}
${TEST_ALGO_SOURCES}
)
