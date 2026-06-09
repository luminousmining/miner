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
endif()
