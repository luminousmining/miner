file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")

target_sources(${BENCH_EXE} PUBLIC
${HEADERS}
${SOURCES}
)

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
