# BLAKE3_CL_PATH is set as a target-level define (not a source-file property):
# ${UNIT_TEST_EXE} is defined in sources/CMakeLists.txt, a different directory
# scope, so set_source_files_properties here would not reach its compile.
if (BUILD_EXE_UNIT_TEST AND BUILD_AMD)
    target_sources(${UNIT_TEST_EXE} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/blake3_kat.cpp
    )
    target_compile_definitions(${UNIT_TEST_EXE} PRIVATE
        "BLAKE3_CL_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/../blake3.cl\""
        "BLAKE3_CRYPTO_CL_PATH=\"${CMAKE_SOURCE_DIR}/sources/algo/crypto/opencl/blake3.cl\""
    )
endif()
