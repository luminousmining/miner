cmake_minimum_required(VERSION 3.22)
project(luminousminer VERSION 0.8)

enable_language(C)
enable_language(CXX)

if (NOT WIN32)
    cmake_policy(SET CMP0135 NEW)
endif()

################################################################################
### OPTION MACROS                                                           ####
################################################################################
option(BUILD_AMD "Build with OpenCL for AMD GPU" ON)
option(BUILD_NVIDIA "Build with CUDA for Nvidia GPU" ON)

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
if (BUILD_NVIDIA)
    set(LIB_CRYPTO_NVIDIA libCryptoNvidia)
    set(LIB_BENCH_NVIDIA libBenchNvidia)
endif()

################################################################################
### CXX FLAGS                                                               ####
################################################################################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_CPPCHECK "cppcheck")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

if (WIN32)
################################################################################
### VISUAL STUDIO                                                           ####
################################################################################
    set(CMAKE_GENERATOR_PLATFORM "x64")
    set(VISUAL_STUDIO_VERSION 143)
    set(CMAKE_SYSTEM_VERSION 10.0.22621.0)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    endif()

    macro(get_WIN32_WINNT version)
    if(CMAKE_SYSTEM_VERSION)
        set(ver ${CMAKE_SYSTEM_VERSION})
        string(REGEX MATCH "^([0-9]+).([0-9])" ver ${ver})
        string(REGEX MATCH "^([0-9]+)" verMajor ${ver})
        # Check for Windows 10, b/c we'll need to convert to hex 'A'.
        if("${verMajor}" MATCHES "10")
        set(verMajor "A")
        string(REGEX REPLACE "^([0-9]+)" ${verMajor} ver ${ver})
        endif("${verMajor}" MATCHES "10")
        # Remove all remaining '.' characters.
        string(REPLACE "." "" ver ${ver})
        # Prepend each digit with a zero.
        string(REGEX REPLACE "([0-9A-Z])" "0\\1" ver ${ver})
        set(${version} "0x${ver}")
    endif(CMAKE_SYSTEM_VERSION)
    endmacro(get_WIN32_WINNT)
    get_win32_winnt(WIN32_VERSION)
    add_definitions(-D_WIN32_WINNT=${WIN32_VERSION})
    add_definitions(/W4)
    add_compile_definitions(NOMINMAX)
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
#        add_definitions(/WX)
    endif()
else()
################################################################################
### CLANG                                                                   ####
################################################################################
    set(CMAKE_C_COMPILER "clang-11" CACHE STRING "C compiler" FORCE)
    set(CMAKE_CXX_COMPILER "clang++-11" CACHE STRING "C++ compiler" FORCE)
    add_compile_options(-W -Wall -Wextra)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g3 -ggdb -v)
    else()
        add_compile_options(-O3)
    endif()
endif()


################################################################################
### BOOST                                                                   ####
################################################################################
if (WIN32)
    set(Boost_ROOT       "C:\\Boost")
    set(BOOST_INCLUDEDIR "C:\\Boost\\include\\boost-1_86")
    set(Boost_COMPILER   -vc${VISUAL_STUDIO_VERSION})
    set(Boost_THREADAPI  win32)
else()
    set(BOOST_INCLUDEDIR "/usr/local/include/boost")
    set(BOOST_LIBRARYDIR "/usr/local/lib")
endif()
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(Boost_DEBUG             ON)
    set(Boost_USE_DEBUG_LIBS    ON)
    set(Boost_USE_DEBUG_RUNTIME ON)
    set(Boost_USE_RELEASE_LIBS  OFF)
    set(Boost_DETAILED_FAILURE_MSG ON)
else()
    set(Boost_DEBUG             OFF)
    set(Boost_USE_DEBUG_LIBS    OFF)
    set(Boost_USE_DEBUG_RUNTIME OFF)
    set(Boost_USE_RELEASE_LIBS  ON)
    set(Boost_DETAILED_FAILURE_MSG ON)
endif()
set(Boost_USE_STATIC_LIBS      ON)
set(Boost_USE_MULTITHREADED    ON)
set(Boost_USE_STATIC_RUNTIME   OFF)


################################################################################
### OPENSSL                                                                 ####
################################################################################
if (WIN32)
    set(OPENSSL_ROOT_DIR "C:\\Program Files\\OpenSSL")
endif()
set(OPENSSL_USE_STATIC_LIBS TRUE)
set(OPEN_SSL_LIBRARIES OpenSSL::SSL OpenSSL::Crypto)


################################################################################
### CUDA                                                                    ####
### https://cmake.org/cmake/help/latest/module/FindCUDA.html                ####
################################################################################
if (BUILD_NVIDIA)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND CUDA_NVCC_FLAGS_DEBUG -G -g -O0)
    else()
        list(APPEND CUDA_NVCC_FLAGS_RELEASE -O3 -DNDEBUG)
    endif()
    ### CUDA GLOBAL INFORMATIONS
    set(CUDA_STANDARD                20)
    set(CUDA_RUNTIME_LIBRARY         STATIC)
    set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "C++ compiler" FORCE)
    ### NVCC FLAGS
    list(APPEND CUDA_NVCC_FLAGS "-std=c++20")
    list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
    list(APPEND CUDA_NVCC_FLAGS "-use_fast_math")
    ### NVCC FLAGS GENCODE
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_60")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_60,code=sm_60") # Pascal
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61") # Pascal
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_62,code=sm_62") # Pascal
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_70,code=sm_70") # Volta
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_75,code=sm_75") # Turing
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_80,code=sm_80") # Ampere
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_86,code=sm_86") # Ampere
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_89,code=sm_89") # Ada(Lovelace)
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_90,code=sm_90") # Hopper
    ### Macros C++
    add_compile_definitions(CUDA_ENABLE)
endif()

################################################################################
### FIND PACKAGE | LIBRARIES                                                ####
################################################################################
find_package(Boost 1.86.0 EXACT COMPONENTS atomic chrono filesystem system json thread serialization program_options)
find_package(OpenCL 2.2 REQUIRED COMPONENTS OpenCL)
if (BUILD_NVIDIA)
    find_package(CUDA COMPONENTS cudart)
endif()
find_package(OpenSSL 1.1.1 REQUIRED COMPONENTS SSL Crypto)
if (UNIX)
    find_package(GnuTLS)
endif()
find_library(OpenCL_LIBRARIES name OpenCL PATHS ${OpenCL_LIBRARY_DIR} PATH_SUFFIXES lib REQUIRED)
if (BUILD_NVIDIA)
    if (WIN32)
        SET(CUDA_TOOLKIT_PATH_SUFFIXES lib64 lib\\x64)
        set(CUDA_NVRTC_BUILTINS_LIBRARY "")
        set(CUDA_NVPTX_LIBRARY "")
        find_library(CUDA_LIBRARY       name cuda  PATHS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES ${CUDA_TOOLKIT_PATH_SUFFIXES} NO_DEFAULT_PATH REQUIRED)
        find_library(CUDA_NVRTC_LIBRARY name nvrtc PATHS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES ${CUDA_TOOLKIT_PATH_SUFFIXES} NO_DEFAULT_PATH REQUIRED)
    else()
        set(CUDA_TOOLKIT_PATH_SUFFIXES lib64 lib/x64 lib64/stubs lib/x64/stubs)
        find_library(CUDA_LIBRARY                name cuda                  PATHS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES ${CUDA_TOOLKIT_PATH_SUFFIXES} NO_DEFAULT_PATH REQUIRED)
        find_library(CUDA_NVRTC_LIBRARY          name nvrtc_static          PATHS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES ${CUDA_TOOLKIT_PATH_SUFFIXES} NO_DEFAULT_PATH REQUIRED)
        find_library(CUDA_NVPTX_LIBRARY          name nvptxcompiler_static  PATHS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES ${CUDA_TOOLKIT_PATH_SUFFIXES} NO_DEFAULT_PATH REQUIRED)
        find_library(CUDA_NVRTC_BUILTINS_LIBRARY name nvrtc-builtins_static PATHS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES ${CUDA_TOOLKIT_PATH_SUFFIXES} NO_DEFAULT_PATH REQUIRED)
    endif()
endif()


################################################################################
### OPENCL                                                                  ####
################################################################################
if (BUILD_AMD)
    if (WIN32)
        set(OpenCL_INCLUDE_DIRS "C:\\OpenCL\\include")
    else()
        set(OpenCL_INCLUDE_DIRS "/usr/local/include")
    endif()
    add_compile_definitions(CL_HPP_ENABLE_EXCEPTIONS=true)
    add_compile_definitions(CL_HPP_TARGET_OPENCL_VERSION=300)
    add_compile_definitions(CL_HPP_MINIMUM_OPENCL_VERSION=200)
    add_compile_definitions(AMD_ENABLE)
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
message("-- C COMPILER       : [${CMAKE_C_COMPILER_ID}] ${CMAKE_C_COMPILER}")
message("-- CXX COMPILER     : [${CMAKE_CXX_COMPILER_ID}] ${CMAKE_CXX_COMPILER}")
message("-- CXX Features     : ${CMAKE_CXX_COMPILE_FEATURES}")
message("-- CXX Version      : ${CMAKE_CXX_STANDARD}")
message("-- CXX LINKER FLAGS : ${CMAKE_EXE_LINKER_FLAGS}")
message("-- CXX FLAGS        : ${CMAKE_CXX_FLAGS}")
message("-- CXX FLAGS D      : ${CMAKE_CXX_FLAGS_DEBUG} ")
message("-- CXX FLAGS R      : ${CMAKE_CXX_FLAGS_RELEASE}")
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
message("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LIBRAIRES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
get_target_property(LIST_LIBS ${MINER_EXE} LINK_LIBRARIES)
string(REPLACE ";" "\n-- " LIST_LIBS "${LIST_LIBS}")
message("-- ${LIST_LIBS}")
message("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
message("=====================================================================")
