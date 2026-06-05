# CMake toolchain for cross-compiling Windows x86-64 binaries from Linux using
# llvm-mingw (https://github.com/mstorsjo/llvm-mingw).
#
# Used as the vcpkg chainload toolchain (VCPKG_CHAINLOAD_TOOLCHAIN_FILE) so that
# BOTH the vcpkg dependencies and the project itself are built with the same
# cross toolchain. The compilers below are expected on PATH (the Docker image
# puts llvm-mingw's bin/ there).

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(TOOLCHAIN_PREFIX x86_64-w64-mingw32)

set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}-clang)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-clang++)
set(CMAKE_RC_COMPILER  ${TOOLCHAIN_PREFIX}-windres)

# Applied to BOTH the vcpkg dependency builds and the project (this is the vcpkg
# chainload toolchain):
#   * _WIN32_WINNT/WINVER = Windows 10 so Boost.Atomic/WinAPI expose WaitOnAddress
#     (requires >= 0x0602); otherwise boost-thread fails to compile.
#   * -pthread so llvm-mingw links winpthreads (otherwise pthread_* are undefined).
set(_lm_flags "-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00 -pthread")
string(APPEND CMAKE_C_FLAGS_INIT   " ${_lm_flags}")
string(APPEND CMAKE_CXX_FLAGS_INIT " ${_lm_flags}")
string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT    " -pthread")
string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " -pthread")

# Look for libraries/headers/packages in the target sysroot (and vcpkg tree),
# but run build tools (ninja, etc.) from the host.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
