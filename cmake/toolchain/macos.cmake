# Xcode ships plain `clang` with no versioned binary — do not force a specific
# compiler name here; CMake will pick up the Xcode toolchain automatically.
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
