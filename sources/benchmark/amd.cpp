#if defined(AMD_ENABLE)

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


std::optional<cl::Device> benchmark::getDevice(uint32_t const index)
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

        size_t const countDevice{ cldevices.size() };
        size_t const totalIndex{ currentIndex + countDevice };
        if (castSize(index) < totalIndex)
        {
            size_t const indexBuffer{ totalIndex - castSize(index) - castSize(1u) };
            cl::Device const& clDevice{ cldevices[indexBuffer] };
            logInfo()
                << "Device ["
                << clDevice.getInfo<CL_DEVICE_BOARD_NAME_AMD>()
                << "] selected!";
            return clDevice;
        }

        currentIndex += cldevices.size();
    }

    return std::nullopt;
}


void benchmark::cleanUpOpenCL(benchmark::PropertiesAmd& properties)
{
    properties.clDevice = nullptr;
    properties.clContext = nullptr;
    properties.clQueue = nullptr;
}


bool benchmark::initializeOpenCL(
        benchmark::PropertiesAmd& properties,
        uint32_t const index)
{
    std::optional<cl::Device> amdDevice{ benchmark::getDevice(index) };
    if (std::nullopt == amdDevice)
    {
        return false;
    }
    properties.clDevice = amdDevice->get();
    properties.clContext = cl::Context(properties.clDevice);
    properties.clQueue = cl::CommandQueue(properties.clContext, properties.clDevice);

    return true;
}

#endif
