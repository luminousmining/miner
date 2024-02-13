set(OUT_ETHASH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/ethash)

file(COPY ethash_dag.cl DESTINATION ${OUT_ETHASH})
file(COPY ethash_result.cl DESTINATION ${OUT_ETHASH})
file(COPY ethash_search.cl DESTINATION ${OUT_ETHASH})
