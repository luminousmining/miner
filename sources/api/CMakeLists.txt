file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")

target_sources(${MINER_EXE} PUBLIC
${HEADERS}
${SOURCES}
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${HEADERS}
${SOURCES}
)


set(SOURCES_API ${HEADERS} ${SOURCES} PARENT_SCOPE)
