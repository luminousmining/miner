set(OUT_AUTOLYKOS_V2 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/autolykos)

file(MAKE_DIRECTORY ${OUT_AUTOLYKOS_V2})

set(AUTOLYKOS_V2_FILES
  autolykos_v2_dag.cl
  autolykos_v2_result.cl
  autolykos_v2_search.cl
  autolykos_v2_var_global.cl
  autolykos_v2_verify.cl
)

foreach(FILE ${AUTOLYKOS_V2_FILES})
  add_custom_command(
    OUTPUT ${OUT_AUTOLYKOS_V2}/${FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${OUT_AUTOLYKOS_V2}/${FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    COMMENT "Copy ${FILE}"
  )
endforeach()

add_custom_target(copy_autolykos_v2_opencl_files ALL
  DEPENDS
    ${OUT_AUTOLYKOS_V2}/autolykos_v2_dag.cl
    ${OUT_AUTOLYKOS_V2}/autolykos_v2_result.cl
    ${OUT_AUTOLYKOS_V2}/autolykos_v2_search.cl
    ${OUT_AUTOLYKOS_V2}/autolykos_v2_var_global.cl
    ${OUT_AUTOLYKOS_V2}/autolykos_v2_verify.cl
)
