set(OUT_COMMON ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/common)

file(MAKE_DIRECTORY ${OUT_COMMON})

set(COMMON_FILES
  be_u32.cuh
  be_u64.cuh
  copy_u4.cuh
  get_lane_id.cuh
  register.cuh
  rotate_byte.cuh
  to_u4.cuh
  to_u64.cuh
  vectorize.cuh
  xor.cuh
)

foreach(FILE ${COMMON_FILES})
  add_custom_command(
    OUTPUT ${OUT_COMMON}/${FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${OUT_COMMON}/${FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    COMMENT "Copy ${FILE}"
  )
endforeach()

add_custom_target(copy_common_cuda_files ALL
  DEPENDS
    ${OUT_COMMON}/be_u32.cuh
    ${OUT_COMMON}/be_u64.cuh
    ${OUT_COMMON}/copy_u4.cuh
    ${OUT_COMMON}/get_lane_id.cuh
    ${OUT_COMMON}/register.cuh
    ${OUT_COMMON}/rotate_byte.cuh
    ${OUT_COMMON}/to_u4.cuh
    ${OUT_COMMON}/to_u64.cuh
    ${OUT_COMMON}/vectorize.cuh
    ${OUT_COMMON}/xor.cuh
)