#include <common/cli/cli.hpp>


uint32_t common::Cli::getOccupancyThreads() const
{
    if (true == contains("threads"))
    {
        return params["threads"].as<uint32_t>();
    }
    return 0u;
}


uint32_t common::Cli::getOccupancyBlocks() const
{
    if (true == contains("blocks"))
    {
        return params["blocks"].as<uint32_t>();
    }
    return 0u;
}


bool common::Cli::isAutoOccupancy() const
{
    bool enable{ false };
    if (true == contains("occupancy"))
    {
        enable = params["occupancy"].as<bool>();
    }
    return enable;
}


uint32_t common::Cli::getInternalLoop() const
{
    uint32_t internalLoop{ 1u };
    if (true == contains("internal_loop"))
    {
        internalLoop = params["internal_loop"].as<uint32_t>();
        if (0u == internalLoop)
        {
            logErr() << "--internal_loop must be greater than 0, reset to default value: 1";
            internalLoop = 1u;
        }
    }
    return internalLoop;
}
