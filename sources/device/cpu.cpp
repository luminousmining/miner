#if defined(CPU_ENABLE)

#include <algorithm>

#include <common/config.hpp>
#include <common/log/log.hpp>
#include <device/cpu.hpp>


bool device::DeviceCpu::initialize()
{
    logInfo() << "Initialize device CPU";
    return true;
}


void device::DeviceCpu::cleanUp()
{
    logInfo() << "Clean up device CPU";
}


uint32_t device::DeviceCpu::getMinimumKernelExecuted() const
{
    // A CPU batch is orders of magnitude slower than a GPU kernel, so the global
    // default (100) is never reached between a pool's frequent job updates and the
    // hashrate would read 0. Cap it low; honour a smaller user value if given.
    constexpr uint32_t CPU_MINIMUM_KERNEL_EXECUTED{ 8u };
    return std::min(device::Device::getMinimumKernelExecuted(), CPU_MINIMUM_KERNEL_EXECUTED);
}

#endif
