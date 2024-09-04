#include <common/cli.hpp>


#if defined(CUDA_ENABLE)
bool common::Cli::isNvidiaEnable() const
{
    bool enable{ true };
    if (true == contains("nvidia"))
    {
        enable = params["nvidia"].as<bool>();
    }
    return enable;
}
#endif


bool common::Cli::isAmdEnable() const
{
    bool enable{ true };
    if (true == contains("amd"))
    {
        enable = params["amd"].as<bool>();
    }
    return enable;
}


bool common::Cli::isCpuEnable() const
{
    bool enable{ true };
    if (true == contains("cpu"))
    {
        enable = params["cpu"].as<bool>();
    }
    return enable;
}