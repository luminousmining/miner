file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")

if (BUILD_EXE_BENCHMARK)
    target_sources(${BENCH_EXE} PUBLIC
        ${HEADERS}
        ${SOURCES}
    )
endif()

set(SOURCES_BENCHMARK ${HEADERS} ${SOURCES} PARENT_SCOPE)

set(OUT_COMMON ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

file(MAKE_DIRECTORY ${OUT_COMMON})

set(BENCHMARK_FILES
    benchmark.py
)

foreach(FILE ${BENCHMARK_FILES})
  add_custom_command(
    OUTPUT ${OUT_COMMON}/${FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${OUT_COMMON}/${FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    COMMENT "Copy ${FILE}"
  )
endforeach()

add_custom_target(copy_benchmark ALL
    DEPENDS
    ${OUT_COMMON}/benchmark.py
)

add_subdirectory(nvidia)
add_subdirectory(amd)
add_subdirectory(cuda)
add_subdirectory(opencl)

set(SOURCES_BENCHMARK_NVIDIA ${SOURCES_BENCHMARK_NVIDIA} PARENT_SCOPE)
set(SOURCES_BENCHMARK_AMD ${SOURCES_BENCHMARK_AMD} PARENT_SCOPE)
set(SOURCES_BENCHMARK_CUDA ${SOURCES_BENCHMARK_CUDA} PARENT_SCOPE)
