file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")

target_sources(${MINER_EXE} PUBLIC
${HEADERS}
${SOURCES}
)

target_sources(${BENCH_EXE} PUBLIC
${HEADERS}
${SOURCES}
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${HEADERS}
${SOURCES}
)

add_subdirectory(cuda)
add_subdirectory(error)
add_subdirectory(kernel_generator)
add_subdirectory(log)
add_subdirectory(opencl)
add_subdirectory(mocker)


set(SOURCES_COMMON ${HEADERS} ${SOURCES} PARENT_SCOPE)
set(SOURCES_COMMON_ERROR ${SOURCES_COMMON_ERROR} PARENT_SCOPE)
set(SOURCES_COMMON_KERNEL_GENERATOR ${SOURCES_COMMON_KERNEL_GENERATOR} PARENT_SCORE)
set(SOURCES_COMMON_LOG ${SOURCES_COMMON_LOG} PARENT_SCOPE)
set(SOURCES_COMMON_MOCKER ${SOURCES_COMMON_MOCKER} PARENT_SCOPE)
