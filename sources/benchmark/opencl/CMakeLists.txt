set(OUT_COMMON ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel)

file(MAKE_DIRECTORY ${OUT_COMMON})

set(BENCHMARK_OPENCL_FILES
    common/init_array.cl
    common/result.cl
    kawpow/kawpow_lm1.cl
    kawpow/sequence_dynamic.cl
)

foreach(FILE ${BENCHMARK_OPENCL_FILES})
  add_custom_command(
    OUTPUT ${OUT_COMMON}/${FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${OUT_COMMON}/${FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    COMMENT "Copy ${FILE}"
  )
endforeach()

add_custom_target(copy_benchmark_opencl ALL
    DEPENDS
    ${OUT_COMMON}/common/init_array.cl
    ${OUT_COMMON}/common/result.cl
    ${OUT_COMMON}/kawpow/kawpow_lm1.cl
    ${OUT_COMMON}/kawpow/sequence_dynamic.cl
)
