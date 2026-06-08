# Standalone CPU-reference build for kHeavyHash (Kaspa).
#
# This is the correctness oracle for the algorithm. It is intentionally
# dependency-free (no Boost/CUDA/OpenCL) so it can be built and tested on a
# minimal toolchain. A later session wires `kheavyhash_reference` and its tests
# into the main `unit_test` target alongside the GPU resolvers.
#
# Build + test in isolation:
#   cmake -S sources/algo/kheavyhash -B build-kheavyhash
#   cmake --build build-kheavyhash
#   ctest --test-dir build-kheavyhash --output-on-failure
cmake_minimum_required(VERSION 3.16)
project(kheavyhash_reference CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(kheavyhash_reference STATIC
    matrix.cpp
    keccak.cpp
    hashers.cpp
    kheavyhash.cpp)

# sources include themselves as <algo/kheavyhash/...>, so the include root is sources/.
target_include_directories(kheavyhash_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_compile_options(kheavyhash_reference PRIVATE -Wall -Wextra)

option(KHEAVYHASH_BUILD_TESTS "Build the kHeavyHash known-answer test suite" ON)
if(KHEAVYHASH_BUILD_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
    add_executable(kheavyhash_unit_test
        tests/matrix_test.cpp
        tests/hashers_test.cpp
        tests/kheavyhash_test.cpp)
    target_include_directories(kheavyhash_unit_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(kheavyhash_unit_test PRIVATE kheavyhash_reference GTest::gtest GTest::gtest_main)
    add_test(NAME kheavyhash_unit_test COMMAND kheavyhash_unit_test)
endif()

# OpenCL kernel KAT harness. Runs the .cl on any OpenCL device the ICD exposes
# (POCL/CPU in the dev image, a real GPU otherwise) and asserts bit-identity
# with the CPU reference. Off by default so the pure-CPU build needs no ICD.
option(KHEAVYHASH_BUILD_OPENCL "Build the kHeavyHash OpenCL known-answer test suite" OFF)
if(KHEAVYHASH_BUILD_OPENCL)
    find_package(GTest REQUIRED)
    find_package(OpenCL REQUIRED)
    enable_testing()
    add_executable(kheavyhash_opencl_test opencl/tests/opencl_kat_test.cpp)
    target_include_directories(kheavyhash_opencl_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_compile_definitions(kheavyhash_opencl_test
        PRIVATE KH_CL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/opencl/kheavyhash.cl")
    target_compile_options(kheavyhash_opencl_test PRIVATE -Wall -Wextra)
    target_link_libraries(kheavyhash_opencl_test
        PRIVATE kheavyhash_reference OpenCL::OpenCL GTest::gtest GTest::gtest_main)
    add_test(NAME kheavyhash_opencl_test COMMAND kheavyhash_opencl_test)
endif()
