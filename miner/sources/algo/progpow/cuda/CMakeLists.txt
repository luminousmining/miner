set(OUT_PROGPOW ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/progpow)

file(COPY firopow_functions.cuh DESTINATION ${OUT_PROGPOW})
file(COPY progpow_functions.cuh DESTINATION ${OUT_PROGPOW})
file(COPY kawpow_functions.cuh DESTINATION ${OUT_PROGPOW})
file(COPY search.cu DESTINATION ${OUT_PROGPOW})
