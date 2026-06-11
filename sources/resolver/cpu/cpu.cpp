#include <common/config.hpp>
#include <resolver/cpu/cpu.hpp>


void resolver::ResolverCpu::overrideOccupancy(uint32_t const defaultThreads, uint32_t const defaultBlocks)
{
    ////////////////////////////////////////////////////////////////////////////
    common::Config const& config{ common::Config::instance() };

    // blocks * threads is the nonce count scanned per executeSync() call. Unlike the
    // GPU paths there is no warp/group-size constraint on the CPU, so honour any
    // user --threads/--blocks verbatim and otherwise fall back to the caller's
    // defaults (kept small so each batch is short: the loop reacts quickly to new
    // jobs and the hashrate counter accumulates enough batches between job resets to
    // display).
    setThreads(std::nullopt != config.occupancy.threads ? *config.occupancy.threads : defaultThreads);
    setBlocks(std::nullopt != config.occupancy.blocks ? *config.occupancy.blocks : defaultBlocks);
}
