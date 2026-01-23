set(OUT_CRYPTO ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/crypto)

file(MAKE_DIRECTORY ${OUT_CRYPTO})

set(CRYPTO_FILES
  fnv1.cuh
  keccak_f800.cuh
  kiss99.cuh
)

foreach(FILE ${CRYPTO_FILES})
  add_custom_command(
    OUTPUT ${OUT_CRYPTO}/${FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${OUT_CRYPTO}/${FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    COMMENT "Copy ${FILE}"
  )
endforeach()

add_custom_target(copy_crypto_cuda_files ALL
  DEPENDS
    ${OUT_CRYPTO}/fnv1.cuh
    ${OUT_CRYPTO}/keccak_f800.cuh
    ${OUT_CRYPTO}/kiss99.cuh
)
