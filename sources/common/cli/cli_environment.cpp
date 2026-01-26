#include <common/cli/cli.hpp>


std::optional<bool> common::Cli::getEnvironmentCudaLazy() const
{
    if (true == contains("env_cuda_lazy"))
    {
        return params["env_cuda_lazy"].as<bool>();
    }
    return std::nullopt;
}


std::optional<std::string> common::Cli::getEnvironmentCudadeviceOrder() const
{
    if (true == contains("env_cuda_device_order"))
    {
        return params["env_cuda_device_order"].as<std::string>();
    }
    return std::nullopt;
}


std::optional<uint32_t> common::Cli::getEnvironmentGpuHeapSize() const
{
    if (true == contains("env_gpu_heap_size"))
    {
        return params["env_gpu_heap_size"].as<uint32_t>();
    }
    return std::nullopt;
}


std::optional<uint32_t> common::Cli::getEnvironmentGpuMaxAllocPercent() const
{
    if (true == contains("env_gpu_max_alloc_percent"))
    {
        return params["env_gpu_max_alloc_percent"].as<uint32_t>();
    }
    return std::nullopt;
}


std::optional<uint32_t> common::Cli::getEnvironmentGpuSingleAllocPercent() const
{
    if (true == contains("env_gpu_single_alloc_percent"))
    {
        return params["env_gpu_single_alloc_percent"].as<uint32_t>();
    }
    return std::nullopt;
}
