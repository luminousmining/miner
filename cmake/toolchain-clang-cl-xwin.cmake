# clang-cl + xwin: cross-compile MSVC-ABI Windows binaries from Linux.
# Used by the x64-windows-clangcl vcpkg triplet and the windows-cross preset.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)
set(CMAKE_CROSSCOMPILING ON)

if(NOT DEFINED XWIN_ROOT)
    if(DEFINED ENV{XWIN_ROOT})
        set(XWIN_ROOT "$ENV{XWIN_ROOT}")
    else()
        set(XWIN_ROOT "/opt/xwin")
    endif()
endif()

set(CMAKE_C_COMPILER   clang-cl)
set(CMAKE_CXX_COMPILER clang-cl)
set(CMAKE_C_COMPILER_TARGET   x86_64-pc-windows-msvc)
set(CMAKE_CXX_COMPILER_TARGET x86_64-pc-windows-msvc)
set(CMAKE_RC_COMPILER  llvm-rc)
set(CMAKE_AR           llvm-lib)
set(CMAKE_MT           llvm-mt)
set(CMAKE_LINKER       lld-link)

set(_inc "/imsvc${XWIN_ROOT}/crt/include /imsvc${XWIN_ROOT}/sdk/include/ucrt /imsvc${XWIN_ROOT}/sdk/include/um /imsvc${XWIN_ROOT}/sdk/include/shared")
set(CMAKE_C_FLAGS_INIT   "${_inc} -Wno-unused-command-line-argument")
set(CMAKE_CXX_FLAGS_INIT "${_inc} -Wno-unused-command-line-argument /EHsc")

# The .rc resource compile path (clang-cl -E preprocess + llvm-rc) does NOT inherit
# the C/CXX /imsvc system includes, so a resource that #include <windows.h> (e.g.
# the OpenCL ICD loader's OpenCL.rc) fails with 'windows.h file not found'. Feed the
# xwin SDK/CRT include dirs to RC explicitly via plain -I (understood by both
# clang-cl -E and llvm-rc; /imsvc is not understood by llvm-rc).
set(CMAKE_RC_FLAGS_INIT "-I${XWIN_ROOT}/sdk/include/um -I${XWIN_ROOT}/sdk/include/shared -I${XWIN_ROOT}/sdk/include/ucrt -I${XWIN_ROOT}/crt/include")

set(_lib "/libpath:${XWIN_ROOT}/crt/lib/x86_64 /libpath:${XWIN_ROOT}/sdk/lib/ucrt/x86_64 /libpath:${XWIN_ROOT}/sdk/lib/um/x86_64")
set(CMAKE_EXE_LINKER_FLAGS_INIT    "${_lib}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_lib}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${_lib}")

# Release-only: force the release CRT so CMake's debug compiler test doesn't pull
# msvcrtd.lib (xwin doesn't splat debug CRT libs by default).
set(CMAKE_TRY_COMPILE_CONFIGURATION Release)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)

# A CMAKE_GENERATOR_PLATFORM=x64 leaks into Ninja try_compiles (e.g. Boost's
# find_dependency(Threads)) -> "Ninja does not support platform specification".
# Clear it so the nested test projects configure. (Same fix as toolchain-mingw.cmake.)
set(CMAKE_GENERATOR_PLATFORM "" CACHE INTERNAL "" FORCE)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
