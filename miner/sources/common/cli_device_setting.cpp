#include <common/cli.hpp>


bool common::Cli::isNvidiaEnable() const
{
    bool enable{ true };
    if (true == contains("nvidia"))
    {
        enable = params["nvidia"].as<bool>();
    }
    return enable;
}


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