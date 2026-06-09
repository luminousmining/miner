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

# Fold the OpenCL kHeavyHash KAT into the main unit_test on the AMD path so the
# bit-identity vectors (reference `search` + the search_lm1/2/3 variants) run on
# the actual GPU. The .cl is loaded at runtime; KH_CL_PATH is the compile-time
# fallback, with an env override and a deploy-relative fallback resolved in the
# test itself, so the same unit_test.exe works in POCL CI and on a real GPU.
if (BUILD_EXE_UNIT_TEST AND BUILD_AMD)
    target_sources(${UNIT_TEST_EXE} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tests/opencl_kat_test.cpp)
    target_include_directories(${UNIT_TEST_EXE} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests
        ${OpenCL_INCLUDE_DIRS})
    target_compile_definitions(${UNIT_TEST_EXE}
        PRIVATE KH_CL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kheavyhash.cl")
endif()
