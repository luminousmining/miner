file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")

if (BUILD_EXE_BENCHMARK)
    target_sources(${BENCH_EXE} PUBLIC
        ${HEADERS}
        ${SOURCES}
    )
endif()

set(SOURCES_BENCHMARK_CUDA ${HEADERS} ${SOURCES} PARENT_SCOPE)
