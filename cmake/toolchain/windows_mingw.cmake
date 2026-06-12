# NOTE: the supported Linux->Windows cross build uses clang-cl + xwin, which
# reports MSVC=true and therefore takes the Visual Studio branch (MSVC ABI,
# single binary co-linking AMD + CUDA). This branch is only reached by a true
# GNU MinGW-w64 g++ compiler (GNU ABI) and is kept as a fallback.
add_compile_definitions(
    _WIN32_WINNT=0x0A00 # Windows 10 API version
    NOMINMAX            # Disable min/max macros from Windows.h
)
add_compile_options(
    -Wall
    -Wextra
    -Wshadow
    -Wdouble-promotion
    -Wformat=2
    -Wnull-dereference
)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g3 -ggdb)
else()
    add_compile_options(-O3)
endif()
# Produce a self-contained .exe: statically link the toolchain runtimes so we
# don't have to ship libc++/libunwind/compiler-rt DLLs next to the binary.
add_link_options(-static)
