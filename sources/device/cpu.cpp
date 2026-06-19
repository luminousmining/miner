#if defined(CPU_ENABLE)

#include <common/config.hpp>
#include <common/custom.hpp>
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
    constexpr uint32_t CPU_MINIMUM_KERNEL_EXECUTED{ 8u };
    return common::max_limit(device::Device::getMinimumKernelExecuted(), CPU_MINIMUM_KERNEL_EXECUTED);
}

#endif
