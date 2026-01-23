set(OUT_PROGPOW ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/progpow)

file(MAKE_DIRECTORY ${OUT_PROGPOW})

set(PROGPOW_FILES
  progpow_result.cl
  evrprogpow_functions.cl
  firopow_functions.cl
  progpow_functions.cl
  kawpow_functions.cl
  meowpow_functions.cl
  progpow.cl
)

foreach(FILE ${PROGPOW_FILES})
  add_custom_command(
    OUTPUT ${OUT_PROGPOW}/${FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${OUT_PROGPOW}/${FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    COMMENT "Copy ${FILE}"
  )
endforeach()

add_custom_target(copy_progpow_opencl_files ALL
  DEPENDS
    ${OUT_PROGPOW}/progpow_result.cl
    ${OUT_PROGPOW}/evrprogpow_functions.cl
    ${OUT_PROGPOW}/firopow_functions.cl
    ${OUT_PROGPOW}/progpow_functions.cl
    ${OUT_PROGPOW}/kawpow_functions.cl
    ${OUT_PROGPOW}/meowpow_functions.cl
    ${OUT_PROGPOW}/progpow.cl
)