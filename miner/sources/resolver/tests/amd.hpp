#pragma once

#include <CL/opencl.hpp>

namespace resolver
{
    namespace tests
    {
        struct Properties
        {
            cl::Device clDevice{ nullptr };
            cl::Context clContext{ nullptr };
            cl::CommandQueue clQueue{ nullptr };
        };

        inline cl::Device getDevice(uint32_t const index)
        {
            std::vector<cl::Device> cldevices{};
            std::vector<cl::Platform> platforms{};
            cl::Platform::get(&platforms);

            for (cl::Platform const& platform : platforms)
            {
                std::string const platformName { platform.getInfo < CL_PLATFORM_NAME > () };
                if (platformName.find("AMD") == std::string::npos)
                {
                    continue;
                }

                platform.getDevices(CL_DEVICE_TYPE_GPU, &cldevices);
                if (false == cldevices.empty() && cldevices.size() > index)
                {
                    auto clDevice{ cldevices.at(index) };
                    logInfo()
                        << "Device ["
                        << clDevice.getInfo<CL_DEVICE_BOARD_NAME_AMD>()
                        << "] selected!";
                    return clDevice;
                }
            }
            return {};
        }

        inline bool initializeOpenCL(resolver::tests::Properties& properties)
        {
            properties.clDevice = resolver::tests::getDevice(0);
            properties.clContext = cl::Context(properties.clDevice);
            properties.clQueue = cl::CommandQueue(properties.clContext, properties.clDevice);

            return true;
        }
    }
}