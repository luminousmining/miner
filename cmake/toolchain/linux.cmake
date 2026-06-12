# macOS has no `clang-15` binary (Xcode ships plain `clang`); only force the
# versioned compiler on non-Apple UNIX.
set(CMAKE_C_COMPILER "clang-15" CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "clang++-15" CACHE STRING "C++ compiler" FORCE)
add_compile_options(
    -Wall              # Enable common warnings
    -Wextra            # Enable additional warnings not covered by -Wall
    -Wshadow           # Warn when variable shadows another
    -Wdouble-promotion # Warn when float is implicitly promoted to double
    -Wformat=2         # Strict format string security checks
    -Wnull-dereference # Warn on potential nullptr dereference
)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g3 -ggdb -v)
else()
    add_compile_options(-O3)
endif()
