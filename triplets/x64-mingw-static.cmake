# Overlay of vcpkg's stock x64-mingw-static triplet that injects the Windows API
# version into dependency builds.
#
# Boost.Atomic / Boost.WinAPI only declare WaitOnAddress / WakeByAddress* (needed
# to compile boost-thread) when _WIN32_WINNT >= 0x0602. The stock mingw triplet
# leaves it unset, so boost-thread fails. VCPKG_CXX_FLAGS is the supported way to
# inject this -- a chainload toolchain's CMAKE_CXX_FLAGS_INIT is ignored by
# vcpkg's port builds. -pthread makes llvm-mingw link winpthreads.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_ENV_PASSTHROUGH PATH)
set(VCPKG_CMAKE_SYSTEM_NAME MinGW)
set(VCPKG_POLICY_DLLS_WITHOUT_LIBS enabled)

# secure_getenv is a glibc extension absent from mingw. The OpenCL-ICD-Loader's
# CMake check false-positives during cross-config and defines HAVE_SECURE_GETENV,
# so it emits secure_getenv() calls that don't link. Map them to getenv (there is
# no setuid distinction on Windows anyway).
set(_mingw_defs "-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00 -pthread -Dsecure_getenv=getenv")
set(VCPKG_CXX_FLAGS "${_mingw_defs}")
set(VCPKG_C_FLAGS   "${_mingw_defs}")
set(VCPKG_LINKER_FLAGS "-pthread")
