set(OUT_PROGPOW ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/progpow)

file(MAKE_DIRECTORY ${OUT_PROGPOW})

set(PROGPOW_FILES
  evrprogpow_functions.cuh
  firopow_functions.cuh
  progpow_functions.cuh
  kawpow_functions.cuh
  meowpow_functions.cuh
  search.cu
)

foreach(FILE ${PROGPOW_FILES})
  add_custom_command(
    OUTPUT ${OUT_PROGPOW}/${FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${OUT_PROGPOW}/${FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    COMMENT "Copy ${FILE}"
  )
endforeach()

add_custom_target(copy_progpow_cuda_files ALL
  DEPENDS
    ${OUT_PROGPOW}/evrprogpow_functions.cuh
    ${OUT_PROGPOW}/firopow_functions.cuh
    ${OUT_PROGPOW}/progpow_functions.cuh
    ${OUT_PROGPOW}/kawpow_functions.cuh
    ${OUT_PROGPOW}/meowpow_functions.cuh
    ${OUT_PROGPOW}/search.cu
)