#include <common/cast.hpp>
#include <common/config.hpp>
#include <common/log/log.hpp>
#include <resolver/cpu/cpu.hpp>


void resolver::ResolverCpu::overrideOccupancy(uint32_t const defaultThreads, uint32_t const defaultBlocks)
{
    ////////////////////////////////////////////////////////////////////////////
    common::Config const& config{ common::Config::instance() };

    // blocks * threads is the nonce count scanned per execute call. Unlike the
    // GPU paths there is no warp/group-size constraint on the CPU, so honour any
    // user --threads/--blocks verbatim and otherwise fall back to the caller's
    // defaults (kept small so each batch is short: the loop reacts quickly to new
    // jobs and the hashrate counter accumulates enough batches between job resets to
    // display).
    uint32_t const threads{ std::nullopt != config.occupancy.threads ? *config.occupancy.threads : defaultThreads };
    uint32_t const blocks{ std::nullopt != config.occupancy.blocks ? *config.occupancy.blocks : defaultBlocks };
    setThreads(threads);
    setBlocks(blocks);

    ////////////////////////////////////////////////////////////////////////////
    // A batch must finish quickly enough that the loop notices a new job (~1/s) and that
    // several batches accumulate between job resets for the hashrate to display. An oversized
    // grid makes each batch long and the miner sluggish to react -- warn the user to dial the
    // grid back rather than silently degrading responsiveness.
    constexpr uint64_t CPU_GRID_WARN_THRESHOLD{ 8ull * 1024ull * 1024ull };
    uint64_t const     count{ castU64(blocks) * castU64(threads) };
    if (CPU_GRID_WARN_THRESHOLD < count)
    {
        logWarn() << "Grid size is too large (threads=" << threads << " blocks=" << blocks << " count=" << count
                  << "). Consider reducing --threads or --blocks to improve "
                  << "batch responsiveness.";
    }
}
