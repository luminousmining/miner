set(OUT_COMMON ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel)

file(MAKE_DIRECTORY ${OUT_COMMON})

set(BENCHMARK_OPENCL_FILES
    common/init_array.cl
    common/result.cl
    autolykos_v2/autolykos_v2_dag.cl
    autolykos_v2/autolykos_v2_search_lm0.cl
    autolykos_v2/autolykos_v2_verify_lm0.cl
    ethash/ethash_lm_0.cl
    ethash/ethash_lm_1.cl
    ethash_light_cache/ethash_light_cache_lm0.cl
    kawpow/kawpow_lm1.cl
    kawpow/kawpow_lm2.cl
    kawpow/kawpow_lm3.cl
    kawpow/kawpow_lm4.cl
    kawpow/sequence_dynamic.cl
    kawpow/sequence_dynamic_local.cl
    progpow/progpow_lm_0.cl
    progpow/progpow_lm_1.cl
    kheavyhash/kHeavyHash_lm0.cl
    kheavyhash/kHeavyHash_lm1.cl
    kheavyhash/kHeavyHash_lm2.cl
    kheavyhash/kHeavyHash_lm3.cl
    kheavyhash/kHeavyHash_lm4.cl
    kheavyhash/kHeavyHash_lm5.cl
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
    ${OUT_COMMON}/autolykos_v2/autolykos_v2_dag.cl
    ${OUT_COMMON}/autolykos_v2/autolykos_v2_search_lm0.cl
    ${OUT_COMMON}/autolykos_v2/autolykos_v2_verify_lm0.cl
    ${OUT_COMMON}/ethash/ethash_lm_0.cl
    ${OUT_COMMON}/ethash/ethash_lm_1.cl
    ${OUT_COMMON}/ethash_light_cache/ethash_light_cache_lm0.cl
    ${OUT_COMMON}/kawpow/kawpow_lm1.cl
    ${OUT_COMMON}/kawpow/kawpow_lm2.cl
    ${OUT_COMMON}/kawpow/kawpow_lm3.cl
    ${OUT_COMMON}/kawpow/kawpow_lm4.cl
    ${OUT_COMMON}/kawpow/sequence_dynamic.cl
    ${OUT_COMMON}/kawpow/sequence_dynamic_local.cl
    ${OUT_COMMON}/progpow/progpow_lm_0.cl
    ${OUT_COMMON}/progpow/progpow_lm_1.cl
    ${OUT_COMMON}/kheavyhash/kHeavyHash_lm0.cl
    ${OUT_COMMON}/kheavyhash/kHeavyHash_lm1.cl
    ${OUT_COMMON}/kheavyhash/kHeavyHash_lm2.cl
    ${OUT_COMMON}/kheavyhash/kHeavyHash_lm3.cl
    ${OUT_COMMON}/kheavyhash/kHeavyHash_lm4.cl
    ${OUT_COMMON}/kheavyhash/kHeavyHash_lm5.cl
)
