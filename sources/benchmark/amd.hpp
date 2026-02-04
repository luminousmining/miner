#pragma once

#if defined(AMD_ENABLE)

#include <optional>

#include <CL/opencl.hpp>

#include <common/log/log.hpp>
#include <common/cast.hpp>


namespace benchmark
{
    struct PropertiesAmd
    {
        cl::Device       clDevice{ nullptr };
        cl::Context      clContext{ nullptr };
        cl::CommandQueue clQueue{ nullptr };
    };

    uint32_t getDeviceCount();
    std::optional<cl::Device> getDevice(uint32_t const index);
    void cleanUpOpenCL(benchmark::PropertiesAmd& properties);
    bool initializeOpenCL(benchmark::PropertiesAmd& properties,
                          uint32_t const index = 0u);
}
#endif
