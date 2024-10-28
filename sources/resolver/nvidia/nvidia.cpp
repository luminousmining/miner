#include <common/config.hpp>
#include <resolver/nvidia/nvidia.hpp>


void resolver::ResolverNvidia::overrideOccupancy(
    uint32_t const defaultThreads,
    uint32_t const defaultBlocks)
{
    ////////////////////////////////////////////////////////////////////////////
    common::Config const& config { common::Config::instance() };

    ////////////////////////////////////////////////////////////////////////////
    if (std::nullopt != config.occupancy.threads)
    {
        uint32_t const threadsCount{ *config.occupancy.threads };
        if (0u == threadsCount % 32u)
        {
            setThreads(threadsCount);
        }
        else
        {
            resolverErr()
                << "Cannot use " << threadsCount
                << " threads. You must define a multiple of 32."
                << " Kernel use 256u by default!";
            setThreads(256u);
        }
    }
    else
    {
        setThreads(defaultThreads);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (std::nullopt != config.occupancy.blocks)
    {
        setBlocks(*config.occupancy.blocks);
    }
    else
    {
        setBlocks(defaultBlocks);
    }

    ////////////////////////////////////////////////////////////////////////////
    resolverDebug()
        << "Occupancy - Threads[" << getThreads()
        << "] Blocks[" << getBlocks() << "]";
}
