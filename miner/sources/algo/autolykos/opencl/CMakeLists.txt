set(OUT_AUTOLYKOS_V2 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/autolykos)

file(COPY autolykos_v2_dag.cl DESTINATION ${OUT_AUTOLYKOS_V2})
file(COPY autolykos_v2_result.cl DESTINATION ${OUT_AUTOLYKOS_V2})
file(COPY autolykos_v2_search.cl DESTINATION ${OUT_AUTOLYKOS_V2})
file(COPY autolykos_v2_var_global.cl DESTINATION ${OUT_AUTOLYKOS_V2})
file(COPY autolykos_v2_verify.cl DESTINATION ${OUT_AUTOLYKOS_V2})