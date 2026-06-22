set(OUT_KHEAVYHASH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/kheavyhash)

file(MAKE_DIRECTORY ${OUT_KHEAVYHASH})

set(KHEAVYHASH_FILES
  kheavyhash.cl
  kheavyhash_result.cl
)

foreach(FILE ${KHEAVYHASH_FILES})
  add_custom_command(
    OUTPUT ${OUT_KHEAVYHASH}/${FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${OUT_KHEAVYHASH}/${FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    COMMENT "Copy ${FILE}"
  )
endforeach()

add_custom_target(copy_kheavyhash_opencl_files ALL
  DEPENDS
    ${OUT_KHEAVYHASH}/kheavyhash.cl
    ${OUT_KHEAVYHASH}/kheavyhash_result.cl
)
