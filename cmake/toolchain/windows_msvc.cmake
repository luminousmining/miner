# Only the Visual Studio multi-config generators honour the platform spec
# (the `-A x64` arg). clang-cl also reports MSVC=true, but its windows-cross
# preset drives Ninja, where a non-empty CMAKE_GENERATOR_PLATFORM leaks into
# FetchContent/try_compile sub-builds (e.g. googletest) and fails with
# "Ninja does not support platform specification". Gate on the generator.
if (CMAKE_GENERATOR MATCHES "Visual Studio")
    set(CMAKE_GENERATOR_PLATFORM "x64")
endif()
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
add_compile_definitions(
    _WIN32_WINNT=${WIN32_VERSION} # Set Windows API version
    NOMINMAX                      # Disable min/max macros from Windows.h
)
add_compile_options(
    /W4 # Warning level 4 (high)
    /MP # Multi-processor compilation
)
