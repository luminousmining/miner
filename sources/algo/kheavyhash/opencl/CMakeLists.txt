# Deploy the kHeavyHash OpenCL kernel next to the binary so the AMD resolver
# can load it at runtime via kernel/kheavyhash/kheavyhash.cl (same scheme as
# the ethash/progpow kernels). The bit-identity KAT harness that also consumes
# this .cl lives in the standalone build (KHEAVYHASH_BUILD_OPENCL), not here.
set(OUT_KHEAVYHASH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/kheavyhash)

file(MAKE_DIRECTORY ${OUT_KHEAVYHASH})

set(KHEAVYHASH_FILES
  kheavyhash.cl
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
)
