# KAT for the shared BLAKE3 OpenCL primitive. Paths are baked as target-level
# defines (${UNIT_TEST_EXE} is defined in a different directory scope, so
# set_source_files_properties here would not reach its compile). Links blake3_ref
# (vendored reference) for the expected values.
if (BUILD_EXE_UNIT_TEST AND BUILD_AMD)
    target_sources(${UNIT_TEST_EXE} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/blake3_kat.cpp
    )
    target_link_libraries(${UNIT_TEST_EXE} PUBLIC blake3_ref)
    target_compile_definitions(${UNIT_TEST_EXE} PRIVATE
        "BLAKE3_CRYPTO_CL_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/../blake3.cl\""
        "BLAKE3_KAT_CL_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/blake3_kat.cl\""
    )

    # Deploy the test wrapper kernel next to the binary so the KAT can run from an
    # installed/cross-built artifact (not just the build tree). The host harness
    # prefers kernel/crypto/tests/blake3_kat.cl, falling back to BLAKE3_KAT_CL_PATH.
    set(OUT_CRYPTO_TESTS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/crypto/tests)
    file(MAKE_DIRECTORY ${OUT_CRYPTO_TESTS})
    add_custom_command(
        OUTPUT ${OUT_CRYPTO_TESTS}/blake3_kat.cl
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/blake3_kat.cl ${OUT_CRYPTO_TESTS}/blake3_kat.cl
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/blake3_kat.cl
        COMMENT "Copy blake3_kat.cl"
    )
    add_custom_target(copy_blake3_kat_opencl_file ALL
        DEPENDS ${OUT_CRYPTO_TESTS}/blake3_kat.cl
    )
endif()
