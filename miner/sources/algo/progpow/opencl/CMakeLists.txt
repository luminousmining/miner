set(OUT_PROGPOW ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/progpow)

file(COPY progpow_result.cl DESTINATION ${OUT_PROGPOW})
file(COPY firopow_seed.cl DESTINATION ${OUT_PROGPOW})
file(COPY progpow_seed.cl DESTINATION ${OUT_PROGPOW})
file(COPY kawpow_seed.cl DESTINATION ${OUT_PROGPOW})
file(COPY progpow.cl DESTINATION ${OUT_PROGPOW})
