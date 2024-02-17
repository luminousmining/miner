set(OUT_PROGPOW ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/progpow)

file(COPY progpow_result.cl DESTINATION ${OUT_PROGPOW})
file(COPY evrprogpow_functions.cl DESTINATION ${OUT_PROGPOW})
file(COPY firopow_functions.cl DESTINATION ${OUT_PROGPOW})
file(COPY progpow_functions.cl DESTINATION ${OUT_PROGPOW})
file(COPY kawpow_functions.cl DESTINATION ${OUT_PROGPOW})
file(COPY progpow.cl DESTINATION ${OUT_PROGPOW})
