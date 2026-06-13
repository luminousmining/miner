set(CUDA_STANDARD                20)
set(CUDA_RUNTIME_LIBRARY         STATIC)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "C++ compiler" FORCE)
list(APPEND CUDA_NVCC_FLAGS "-std=c++20")
list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
list(APPEND CUDA_NVCC_FLAGS "-use_fast_math")
list(APPEND CUDA_NVCC_FLAGS "-Xptxas -v")
# Parallelize per-arch device codegen across CPU threads; speeds the multi-arch
# release fat-binary build and is a no-op for single-arch CI builds.
list(APPEND CUDA_NVCC_FLAGS "-t0")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND CUDA_NVCC_FLAGS_DEBUG -G -g -O0)
else()
    list(APPEND CUDA_NVCC_FLAGS_RELEASE -O3 -DNDEBUG)
endif()

if (NVCC_ARCH_COMPUTE)
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_${NVCC_ARCH_COMPUTE}")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_${NVCC_ARCH_COMPUTE},code=sm_${NVCC_ARCH_COMPUTE}")
else()
    list(APPEND CUDA_NVCC_FLAGS
        # ============================================================
        # ============================================================

        "-gencode arch=compute_75,code=sm_75"     # Turing — GTX 1660, RTX 2060, RTX 2080

        "-gencode arch=compute_86,code=sm_86"     # Ampere RTX — RTX 3060, 3070, 3080, 3090
        "-gencode arch=compute_89,code=sm_89"     # Ada Lovelace — RTX 4060, 4070, 4080, 4090

        "-gencode arch=compute_120,code=sm_120"   # Blackwell CC 12.0 — RTX 50xx (expected)
        "-gencode arch=compute_121,code=sm_121"   # Blackwell CC 12.1 — RTX 50xx refresh

        # ============================================================
        # ============================================================

        "-gencode arch=compute_80,code=sm_80"     # Ampere A100 — A100 40GB / 80GB

        "-gencode arch=compute_90,code=sm_90"     # Hopper — H100 SXM / PCIe
        "-gencode arch=compute_90a,code=sm_90a"   # Hopper optimized — H100 NVL (HPC / AI)

        "-gencode arch=compute_100,code=sm_100"   # Blackwell CC 10.0 — B100
        "-gencode arch=compute_100a,code=sm_100a" # Blackwell CC 10.0a — B100 NVL

        "-gencode arch=compute_103,code=sm_103"   # Blackwell CC 10.3 — B200
        "-gencode arch=compute_103a,code=sm_103a" # Blackwell CC 10.3a — B200 NVL

        "-gencode arch=compute_110,code=sm_110"   # Blackwell CC 11.0 — intermediate datacenter gen
        "-gencode arch=compute_110a,code=sm_110a" # Blackwell CC 11.0a — NVL / HPC variants
    )
endif()
add_compile_definitions(CUDA_ENABLE)

find_package(CUDA COMPONENTS cudart)

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
