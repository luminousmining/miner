# The KAT loads the kernels deployed next to the binary (kernel/crypto/blake3.cl +
# kernel/blake3/blake3.cl, copied by the production opencl CMakeLists), the same way
# the other OpenCL unit tests resolve their .cl.
if (BUILD_EXE_UNIT_TEST AND BUILD_AMD)
    target_sources(${UNIT_TEST_EXE} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/blake3_kat.cpp
    )
endif()
