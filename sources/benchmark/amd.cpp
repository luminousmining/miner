#include <CL/opencl.hpp>

#include <common/custom.hpp>
#include <benchmark/amd.hpp>
#include <common/log/log.hpp>
#include <common/cast.hpp>


uint32_t benchmark::getDeviceCount()
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


cl::Device benchmark::getDevice(uint32_t const index)
{
    std::vector<cl::Device> cldevices{};
    std::vector<cl::Platform> platforms{};
    cl::Platform::get(&platforms);

    size_t currentIndex{ 0u };
    for (cl::Platform const& platform : platforms)
    {
        std::string const platformName { platform.getInfo<CL_PLATFORM_NAME>() };
        if (platformName.find("AMD") == std::string::npos)
        {
            continue;
        }

        platform.getDevices(CL_DEVICE_TYPE_GPU, &cldevices);

        if (true == cldevices.empty())
        {
            continue;
        }

        size_t const countDevice{ castU32(cldevices.size()) };
        size_t const totalIndex{ currentIndex + countDevice };
        if (index < totalIndex)
        {
            size_t const indexBuffer{ totalIndex - index - 1u };
            auto const& clDevice{ cldevices.at(indexBuffer) };
            logInfo()
                << "Device ["
                << clDevice.getInfo<CL_DEVICE_BOARD_NAME_AMD>()
                << "] selected!";
            return clDevice;
        }

        currentIndex += cldevices.size();
    }

    return {};
}


void benchmark::cleanUpOpenCL(benchmark::PropertiesAmd& properties)
{
    properties.clDevice = nullptr;
    properties.clContext = nullptr;
    properties.clQueue = nullptr;
}


bool benchmark::initializeOpenCL(
        benchmark::PropertiesAmd& properties,
        uint32_t index)
{
    properties.clDevice = benchmark::getDevice(index);
    properties.clContext = cl::Context(properties.clDevice);
    properties.clQueue = cl::CommandQueue(properties.clContext, properties.clDevice);

    return true;
}
