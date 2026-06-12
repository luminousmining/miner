if (BUILD_AMD)
    find_package(OpenCL COMPONENTS OpenCL)
    add_compile_definitions(
        CL_HPP_ENABLE_EXCEPTIONS=true     # Enable C++ exceptions in OpenCL C++ wrapper
        CL_HPP_TARGET_OPENCL_VERSION=300  # Target OpenCL 3.0
        CL_HPP_MINIMUM_OPENCL_VERSION=200 # Minimum OpenCL 2.0 required
        AMD_ENABLE                        # Enable AMD-specific features
    )
endif()
