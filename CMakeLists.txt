cmake_minimum_required(VERSION 3.22)
project(luminousminer VERSION 0.10)

enable_language(C)
enable_language(CXX)

if (NOT WIN32)
    cmake_policy(SET CMP0135 NEW)
endif()


################################################################################
### COMPUTE BACKENDS
################################################################################
option(BUILD_AMD    "Build with OpenCL for AMD GPU"          ON)
option(BUILD_CPU    "Build with OpenMP for CPU (Intel/AMD)"  ON)
option(BUILD_NVIDIA "Build with CUDA for Nvidia GPU"         ON)


################################################################################
### EXECUTABLES
################################################################################
option(BUILD_EXE_MINER     "Build miner executable"      ON)
option(BUILD_EXE_UNIT_TEST "Build unit test executable"  ON)
option(BUILD_EXE_BENCHMARK "Build benchmark executable"  ON)


################################################################################
### BUILD SYSTEM
################################################################################
option(USE_VCPKG      "Resolve dependencies via vcpkg toolchain instead of hard-coded paths" OFF)
option(USE_CLANG_CUDA "Compile .cu with clang (not nvcc/FindCUDA) for Windows cross-compile" OFF)


################################################################################
### DEVELOPER TOOLS
################################################################################
option(TOOL_TRACE_MEMORY "Tool for develloper, trace memory allocation" OFF)
option(TOOL_MOCKER       "Tool for develloper, insert devices fake"      OFF)
option(TOOL_ANALYZER     "Tool for develloper, analyzer source code"     OFF)

################################################################################
### DEFAULT BUILD TYPE                                                      ####
################################################################################
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Debug")
    set(CMAKE_BUILD_TYPE "Debug" CACHE PATH "Build Type" FORCE)
endif()

################################################################################
### PROJECTS NAME                                                           ####
################################################################################
set(MINER_EXE miner)
set(BENCH_EXE benchmark)
set(UNIT_TEST_EXE unit_test)
if (BUILD_NVIDIA AND (BUILD_EXE_MINER OR BUILD_EXE_UNIT_TEST))
    set(LIB_CRYPTO_NVIDIA libCryptoNvidia)
endif()
if (BUILD_NVIDIA AND BUILD_EXE_BENCHMARK)
    set(LIB_BENCH_NVIDIA libBenchNvidia)
endif()

################################################################################
### CXX FLAGS                                                               ####
################################################################################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)


################################################################################
### TOOLS                                                                   ####
################################################################################
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tools/format.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tools/analyzer.cmake)


################################################################################
### TOOLCHAIN                                                               ####
################################################################################
if (WIN32)
    if (MSVC)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchain/windows_msvc.cmake)
    else()
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchain/windows_mingw.cmake)
    endif()
elseif (UNIX AND NOT APPLE)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchain/linux.cmake)
else()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchain/macos.cmake)
endif()


################################################################################
### DEPENDENCIES                                                            ####
################################################################################
if (USE_VCPKG)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/vcpkg/boost.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/vcpkg/openssl.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/vcpkg/opencl.cmake)
else()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/system/boost.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/system/openssl.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/system/opencl.cmake)
endif()
if (BUILD_NVIDIA)
    if (USE_CLANG_CUDA)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/cuda/clang.cmake)
    else()
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/cuda/nvcc.cmake)
    endif()
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/gnutls.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/cpu.cmake)


################################################################################
### DEVELOPER FLAGS                                                         ####
################################################################################
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tools/trace_memory.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tools/mocker.cmake)
if (TOOL_TRACE_MEMORY OR TOOL_MOCKER)
    add_compile_definitions(TOOLS_ENABLE)
endif()


################################################################################
### PROJECTS                                                                ####
################################################################################
add_subdirectory(sources)


################################################################################
### INFORMATIONS                                                            ####
################################################################################
message("=====================================================================")
message("*********************************************************************")
message("-- Platform         : ${CMAKE_SYSTEM_INFO_FILE}")
message("-- Build Type       : ${CMAKE_BUILD_TYPE}")
message("-- Runtime Output   : ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message("-- Library Output   : ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message("*********************************************************************")
message("-- Miner            : ${MINER_EXE}")
message("-- Benchmark        : ${BENCH_EXE}")
message("-- Unit Test        : ${UNIT_TEST_EXE}")
if (BUILD_NVIDIA)
message("-- LibCryptoNvidia  : ${LIB_CRYPTO_NVIDIA}")
message("-- LibBenchNvidia   : ${LIB_BENCH_NVIDIA}")
endif()
message("*********************************************************************")
message("-- C COMPILER       : [${CMAKE_C_COMPILER_ID}] ${CMAKE_C_COMPILER}")
message("-- CXX COMPILER     : [${CMAKE_CXX_COMPILER_ID}] ${CMAKE_CXX_COMPILER}")
message("-- CXX Features     : ${CMAKE_CXX_COMPILE_FEATURES}")
message("-- CXX Version      : ${CMAKE_CXX_STANDARD}")
message("-- CXX LINKER FLAGS : ${CMAKE_EXE_LINKER_FLAGS}")
message("-- CXX FLAGS        : ${CMAKE_CXX_FLAGS}")
message("-- CXX FLAGS D      : ${CMAKE_CXX_FLAGS_DEBUG} ")
message("-- CXX FLAGS R      : ${CMAKE_CXX_FLAGS_RELEASE}")
message("*********************************************************************")
message("-- CPP Checker      : ${CPPCHECK_EXECUTABLE}")
message("-- Clang Tidy       : ${CLANG_TIDY_EXECUTABLE}")
message("*********************************************************************")
message("-- Boost Version    : ${Boost_LIB_VERSION}")
message("-- Boost Include    : ${BOOST_INCLUDEDIR}")
message("-- Boost DirectoryL : ${Boost_LIBRARY_DIRS}")
message("-- Boost Libraries  : ${Boost_LIBRARIES}")
message("*********************************************************************")
message("-- OpenSSL Root Dir : ${OPENSSL_ROOT_DIR}")
message("-- OpenSSL Version  : ${OPENSSL_VERSION}")
message("-- OpenSSL Include  : ${OPENSSL_INCLUDE_DIR}")
message("-- OpenSSL Library  : ${OPEN_SSL_LIBRARIES}")
message("*********************************************************************")
message("-- BUILD_AMD        : ${BUILD_AMD}")
if (BUILD_AMD)
message("-- OPENCL           : ${OpenCL_VERSION_STRING}")
message("-- OPENCL Include   : ${OpenCL_INCLUDE_DIRS}")
message("-- OPENCL Libraries : ${OpenCL_LIBRARIES}")
endif()
message("*********************************************************************")
message("-- BUILD_CPU        : ${BUILD_CPU}")
message("*********************************************************************")
message("-- BUILD_NVIDIA     : ${BUILD_NVIDIA}")
if (BUILD_NVIDIA)
message("-- CUDA Version     : ${CUDA_VERSION}")
message("-- CUDA Flags       : ${CUDA_NVCC_FLAGS};${CUDA_NVCC_FLAGS_DEBUG}${CUDA_NVCC_FLAGS_RELEASE}")
message("-- CUDA TOOLKIT     : ${CUDA_TOOLKIT_ROOT_DIR}")
message("-- CUDA_LIBRARIES   : ${CUDA_LIBRARIES}")
message("-- CUDA LIBRARY     : ${CUDA_LIBRARY}")
message("-- CUDA NVPTX       : ${CUDA_NVPTX_LIBRARY}")
message("-- CUDA NVRTC       : ${CUDA_NVRTC_LIBRARY}")
message("-- CUDA BUILTINS    : ${CUDA_NVRTC_BUILTINS_LIBRARY}")
message("-- NVRTC LIBRARY    : ${CUDA_NVRTC_LIBRARY}")
endif()
message("*********************************************************************")
message("-- BUILD NVIDIA     : ${BUILD_NVIDIA}")
message("-- BUILD AMD        : ${BUILD_AMD}")
message("-- BUILD CPU        : ${BUILD_CPU}")
message("*********************************************************************")
message("-- BUILD MINER      : ${BUILD_EXE_MINER}")
message("-- BUILD BENCHMARK  : ${BUILD_EXE_BENCHMARK}")
message("-- BUILD UNIT TEST  : ${BUILD_EXE_UNIT_TEST}")
message("*********************************************************************")
message("*********************************************************************")
message("-- TOOL_TRACE_MEMORY : ${TOOL_TRACE_MEMORY}")
message("-- TOOL_MOCKER       : ${TOOL_MOCKER}")
message("-- TOOL_ANALYZER:    : ${TOOL_ANALYZER}")
if (TOOL_ANALYZER)
message("-- CLANG FORMAT      : ${CLANG_FORMAT_EXECUTABLE}")
endif()
message("*********************************************************************")
message("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LIBRAIRES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
if (BUILD_EXE_MINER)
    get_target_property(LIST_LIBS ${MINER_EXE} LINK_LIBRARIES)
    string(REPLACE ";" "\n-- " LIST_LIBS "${LIST_LIBS}")
    message("-- ${LIST_LIBS}")
endif()
message("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
message("=====================================================================")
