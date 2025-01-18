#include <common/cli.hpp>


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
