# Clang CUDA-language cross. CMake cannot drive clang+CUDA when targeting
# Windows (issue #20776), so the .cu units are compiled by clang directly via
# add_custom_command in sources/CMakeLists.txt. These vars feed those commands.
# This path is wired for the windows-cross Docker image; the staging
# env vars must be present (fail fast with a clear message if configured bare).
if (NOT DEFINED ENV{CUDA_WIN_LIB})
    message(FATAL_ERROR
        "USE_CLANG_CUDA expects the windows-cross Docker environment "
        "(CUDA_WIN_LIB must point at the staged CUDA import libs). Build via "
        "docker/Dockerfile.windows-cross, not a bare configure.")
endif()
if (DEFINED ENV{XWIN_ROOT})
    set(_xwin "$ENV{XWIN_ROOT}")
else()
    set(_xwin "/opt/xwin")
endif()
if (DEFINED ENV{CUDA_HOME})
    set(CLANG_CUDA_PATH "$ENV{CUDA_HOME}")
else()
    set(CLANG_CUDA_PATH "/usr/local/cuda")
endif()
set(CLANG_CUDA_COMPILER "/opt/llvm-mingw/bin/clang++")
# Mirror the nvcc path: NVCC_ARCH_COMPUTE narrows the build to a single GPU
# target for fast dev iteration; unset builds the full default arch set.
if (NVCC_ARCH_COMPUTE)
    set(CLANG_CUDA_GPU_ARCHS sm_${NVCC_ARCH_COMPUTE})
else()
    set(CLANG_CUDA_GPU_ARCHS sm_75 sm_86 sm_89 sm_120 sm_121)
endif()
set(CLANG_CUDA_XWIN_INC
    "-I${_xwin}/crt/include" "-I${_xwin}/sdk/include/ucrt"
    "-I${_xwin}/sdk/include/um" "-I${_xwin}/sdk/include/shared")
# FindCUDA is not run on this path; host TUs that include cuda_runtime.h need this.
set(CUDA_INCLUDE_DIRS "${CLANG_CUDA_PATH}/include")
add_compile_definitions(CUDA_ENABLE)

# Compile each .cu with clang (-x cuda) into a Windows-MSVC object and archive
# them into a STATIC lib. Used in place of cuda_add_library on the cross path.
function(lm_clang_cuda_library target)
    set(_archflags "")
    foreach(a ${CLANG_CUDA_GPU_ARCHS})
        list(APPEND _archflags "--cuda-gpu-arch=${a}")
    endforeach()
    set(_objs "")
    foreach(src ${ARGN})
        # Key the object on the full relative path (sanitized), not just the
        # basename, so two .cu with the same filename in different dirs can't collide.
        string(REGEX REPLACE "[/\\.]" "_" _nm "${src}")
        set(_obj "${CMAKE_CURRENT_BINARY_DIR}/${target}_${_nm}.obj")
        add_custom_command(
            OUTPUT "${_obj}"
            # Release device codegen: -O3 -DNDEBUG matches the nvcc release path
            # (CUDA_NVCC_FLAGS_RELEASE). Without it clang defaults to -O0 and the
            # mining kernels -- the hot path -- ship unoptimized.
            COMMAND "${CLANG_CUDA_COMPILER}"
                    --target=x86_64-pc-windows-msvc
                    -x cuda
                    "${CMAKE_CURRENT_SOURCE_DIR}/${src}"
                    -c
                    -o "${_obj}"
                    --cuda-path=${CLANG_CUDA_PATH}
                    ${_archflags}
                    -O3
                    -DNDEBUG
                    -Wno-unknown-cuda-version
                    -std=c++20
                    -fms-compatibility
                    -fms-extensions
                    -fms-runtime-lib=dll
                    -Wno-unknown-attributes
                    -D_MT
                    -D_DLL
                    ${CLANG_CUDA_XWIN_INC}
                    "-I${CMAKE_SOURCE_DIR}/sources"
                    "-I${Boost_INCLUDE_DIR}"
                    "-I${CUDA_INCLUDE_DIRS}"
                    -D__LIB_CUDA
                    -DCUDA_ENABLE
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${src}"
            VERBATIM)
        list(APPEND _objs "${_obj}")
        set_source_files_properties("${_obj}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    endforeach()
    add_library(${target} STATIC ${_objs})
    set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

# Windows CUDA import libs from the redist, staged by the Dockerfile.
set(_cuda_win_lib "$ENV{CUDA_WIN_LIB}")
find_library(CUDA_LIBRARY        NAMES cuda   PATHS "${_cuda_win_lib}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED)
find_library(CUDA_NVRTC_LIBRARY  NAMES nvrtc  PATHS "${_cuda_win_lib}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED)
find_library(CUDA_CUDART_LIBRARY NAMES cudart PATHS "${_cuda_win_lib}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED)
# The native FindCUDA path auto-links cudart; here fold it into CUDA_LIBRARY so
# every `${CUDA_LIBRARY}` link site also pulls the CUDA runtime (cudaGetDeviceCount, ...).
set(CUDA_LIBRARY "${CUDA_LIBRARY};${CUDA_CUDART_LIBRARY}")
