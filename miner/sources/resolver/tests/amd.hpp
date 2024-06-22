#pragma once

#include <CL/opencl.hpp>
#include <common/log/log.hpp>


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

        inline uint32_t getDeviceCount()
        {
            uint32_t count { 0u };
            std::vector<cl::Device> cldevices{};
            std::vector<cl::Platform> platforms{};
            cl::Platform::get(&platforms);

            for (cl::Platform const& platform : platforms)
            {
                std::string const platformName { platform.getInfo<CL_PLATFORM_NAME>() };
                if (platformName.find("AMD") == std::string::npos)
                {
                    continue;
                }

                platform.getDevices(CL_DEVICE_TYPE_GPU, &cldevices);
                count += castU32(cldevices.size());
            }

            return count;
        }

        inline cl::Device getDevice(uint32_t const index)
        {
            std::vector<cl::Device> cldevices{};
            std::vector<cl::Platform> platforms{};
            cl::Platform::get(&platforms);

            for (cl::Platform const& platform : platforms)
            {
                std::string const platformName { platform.getInfo<CL_PLATFORM_NAME>() };
                if (platformName.find("AMD") == std::string::npos)
                {
                    continue;
                }

                platform.getDevices(CL_DEVICE_TYPE_GPU, &cldevices);
                if (false == cldevices.empty())
                {
                    if (cldevices.size() > index)
                    {
                        auto clDevice{ cldevices.at(index) };
                        logInfo()
                            << "Device ["
                            << clDevice.getInfo<CL_DEVICE_BOARD_NAME_AMD>()
                            << "] selected!";
                        return clDevice;
                    }
                }

                logErr() << "Platform AMD does not gpu index[" << index << "]";
                break;
            }

            return {};
        }

        inline bool initializeOpenCL(
            resolver::tests::Properties& properties,
            uint32_t index = 0u)
        {
            properties.clDevice = resolver::tests::getDevice(index);
            properties.clContext = cl::Context(properties.clDevice);
            properties.clQueue = cl::CommandQueue(properties.clContext, properties.clDevice);

            return true;
        }
    }
}
