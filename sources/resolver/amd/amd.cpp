#include <common/cast.hpp>
#include <common/config.hpp>
#include <resolver/amd/amd.hpp>


void resolver::ResolverAmd::setDevice(
    cl::Device* const device)
{
    clDevice = device;
}


void resolver::ResolverAmd::setContext(
    cl::Context* const context)
{
    clContext = context;
}


void resolver::ResolverAmd::setQueue(
    cl::CommandQueue* const queue)
{
    clQueue[0] = &queue[0];
    clQueue[1] = &queue[1];
}


uint32_t resolver::ResolverAmd::getMaxGroupSize() const
{
    return castU32(clDevice->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
}


void resolver::ResolverAmd::overrideOccupancy(
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
                << " Kernel use 256 threads by default!";
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
        uint32_t const blocksCount{ *config.occupancy.blocks };

        if (   0u == blocksCount % 32u
            && getMaxGroupSize() >= blocksCount)
        {
            setBlocks(blocksCount);
        }
        else
        {
            resolverErr()
                << "Cannot use " << blocksCount << "block."
                << " You must define a multiple of 32 or less than or equal to " << getMaxGroupSize() << "."
                << " Kernel use " << getMaxGroupSize() << " blocks by default!";
            setThreads(getMaxGroupSize());
        }
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