if (BUILD_AMD)
    if (WIN32)
        set(OpenCL_INCLUDE_DIRS "C:\\OpenCL\\include")
    else()
        set(OpenCL_INCLUDE_DIRS "/usr/local/include")
    endif()
    find_package(OpenCL COMPONENTS OpenCL)
    # find_package may set only the include dir on hard-coded-path builds;
    # fall back to a manual library search.
    if (NOT OpenCL_LIBRARIES)
        find_library(OpenCL_LIBRARIES NAMES OpenCL PATHS ${OpenCL_LIBRARY_DIR} PATH_SUFFIXES lib REQUIRED)
    endif()
    add_compile_definitions(
        CL_HPP_ENABLE_EXCEPTIONS=true     # Enable C++ exceptions in OpenCL C++ wrapper
        CL_HPP_TARGET_OPENCL_VERSION=300  # Target OpenCL 3.0
        CL_HPP_MINIMUM_OPENCL_VERSION=200 # Minimum OpenCL 2.0 required
        AMD_ENABLE                        # Enable AMD-specific features
    )
endif()
