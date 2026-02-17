#include <CL/opencl.hpp>

#include <algo/keccak.hpp>
#include <algo/ethash/ethash.hpp>
#include <common/cast.hpp>
#include <common/chrono.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <common/error/opencl_error.hpp>
#include <resolver/amd/etchash.hpp>


resolver::ResolverAmdEtchash::ResolverAmdEtchash():
    resolver::ResolverAmdEthash()
{
    algorithm = algo::ALGORITHM::ETHASH;
}


bool resolver::ResolverAmdEtchash::updateContext(
    stratum::StratumJobInfo const& jobInfo)
{
    ///////////////////////////////////////////////////////////////////////////
    algo::ethash::ContextGenerator::instance().build
    (
        algorithm,
        context,
        jobInfo.epoch,
        algo::ethash::EIP1099_MAX_EPOCH_NUMBER,
        dagCountItemsGrowth,
        dagCountItemsInit,
        lightCacheCountItemsGrowth,
        lightCacheCountItemsInit,
        true /*config.deviceAlgorithm.ethashBuildLightCacheCPU*/
    );

    if (   context.lightCache.numberItem == 0ull
        || context.lightCache.size == 0ull
        || context.dagCache.numberItem == 0ull
        || context.dagCache.size == 0ull)
    {
        resolverErr()
            << "\n"
            << "=========================================================================" << "\n"
            << "context.lightCache.numberItem: " << context.lightCache.numberItem << "\n"
            << "context.lightCache.size: " << context.lightCache.size << "\n"
            << "context.dagCache.numberItem: " << context.dagCache.numberItem << "\n"
            << "context.dagCache.size: " << context.dagCache.size << "\n"
            << "=========================================================================" << "\n"
            ;
        return false;
    }

    uint64_t const totalMemoryNeeded{ (context.dagCache.size + context.lightCache.size) };
    if (   0ull != deviceMemoryAvailable
        && totalMemoryNeeded >= deviceMemoryAvailable)
    {
        resolverErr()
            << "Device have not memory size available."
            << " Needed " << totalMemoryNeeded << ", memory available " << deviceMemoryAvailable;
        return false;
    }

    return true;
}
