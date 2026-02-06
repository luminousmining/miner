set(OUT_COMMON ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/common)

file(MAKE_DIRECTORY ${OUT_COMMON})

set(COMMON_FILES
  cross_lane.cl
  debug.cl
  grid.cl
  rotate_byte.cl
  to_u4.cl
  xor.cl
)

foreach(FILE ${COMMON_FILES})
  add_custom_command(
    OUTPUT ${OUT_COMMON}/${FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${OUT_COMMON}/${FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    COMMENT "Copy ${FILE}"
  )
endforeach()

add_custom_target(copy_common_opencl_files ALL
  DEPENDS
    ${OUT_COMMON}/cross_lane.cl
    ${OUT_COMMON}/debug.cl
    ${OUT_COMMON}/grid.cl
    ${OUT_COMMON}/rotate_byte.cl
    ${OUT_COMMON}/to_u4.cl
    ${OUT_COMMON}/xor.cl
)
