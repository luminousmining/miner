set(OUT_ETHASH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/ethash)

file(MAKE_DIRECTORY ${OUT_ETHASH})

set(ETHASH_FILES
  ethash_dag.cl
  ethash_result.cl
  ethash_search.cl
)

foreach(FILE ${ETHASH_FILES})
  add_custom_command(
    OUTPUT ${OUT_ETHASH}/${FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${OUT_ETHASH}/${FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    COMMENT "Copy ${FILE}"
  )
endforeach()

add_custom_target(copy_ethash_opencl_files ALL
  DEPENDS
    ${OUT_ETHASH}/ethash_dag.cl
    ${OUT_ETHASH}/ethash_result.cl
    ${OUT_ETHASH}/ethash_search.cl
)
