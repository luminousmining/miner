# Deploy the Blake3 OpenCL kernel next to the binary (kernel/blake3/blake3.cl),
# loaded at runtime by the AMD resolver. The same .cl is consumed by the POCL
# KAT below, so there is no second copy to drift.
set(OUT_BLAKE3 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/blake3)

file(MAKE_DIRECTORY ${OUT_BLAKE3})

set(BLAKE3_FILES
  blake3.cl
)

foreach(FILE ${BLAKE3_FILES})
  add_custom_command(
    OUTPUT ${OUT_BLAKE3}/${FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${OUT_BLAKE3}/${FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    COMMENT "Copy ${FILE}"
  )
endforeach()

add_custom_target(copy_blake3_opencl_files ALL
  DEPENDS
    ${OUT_BLAKE3}/blake3.cl
)

# POCL/OpenCL bit-identity KAT, compiled into unit_test when AMD is enabled.
if (BUILD_EXE_UNIT_TEST AND BUILD_AMD)
    add_subdirectory(tests)
endif()
