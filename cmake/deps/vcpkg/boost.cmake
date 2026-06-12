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

# vcpkg ships Boost as a CONFIG package; its exact version follows the vcpkg
# baseline (see vcpkg.json) rather than being pinned to 1.91.0 here.
# Boost.System is header-only as of 1.91 (no libboost_system to find), so it
# is omitted from the component list -- the headers come in transitively.
find_package(Boost CONFIG REQUIRED COMPONENTS atomic chrono filesystem json thread serialization program_options)
if (NOT Boost_LIBRARIES)
    set(Boost_LIBRARIES
        Boost::atomic
        Boost::chrono
        Boost::filesystem
        Boost::json
        Boost::thread
        Boost::serialization
        Boost::program_options)
endif()
# The legacy FindCUDA/nvcc path (sources/CMakeLists.txt feeds it
# ${Boost_INCLUDE_DIR}) does NOT pick up include dirs from imported targets,
# and CONFIG mode leaves the plain Boost_INCLUDE_DIR empty -> nvcc can't find
# boost headers. Derive it from the Boost::headers target / Boost_INCLUDE_DIRS.
if (NOT Boost_INCLUDE_DIR)
    if (Boost_INCLUDE_DIRS)
        set(Boost_INCLUDE_DIR "${Boost_INCLUDE_DIRS}")
    elseif (TARGET Boost::headers)
        get_target_property(Boost_INCLUDE_DIR Boost::headers INTERFACE_INCLUDE_DIRECTORIES)
    endif()
endif()
