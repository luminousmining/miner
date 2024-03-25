#include <algo/keccak.hpp>
#include <algo/ethash/ethash.hpp>
#include <algo/ethash/cuda/ethash.cuh>
#include <common/cast.hpp>
#include <common/chrono.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <resolver/nvidia/etchash.hpp>


bool resolver::ResolverNvidiaEtchash::updateContext(
    stratum::StratumJobInfo const& jobInfo)
{
    algo::ethash::initializeDagContext(context,
                                       jobInfo.epoch,
                                       algo::ethash::EIP1099_MAX_EPOCH_NUMBER,
                                       algo::ethash::DAG_COUNT_ITEMS_GROWTH,
                                       algo::ethash::DAG_COUNT_ITEMS_INIT);

    if (   context.lightCache.numberItem == 0ull
        || context.lightCache.size == 0ull
        || context.dagCache.numberItem == 0ull
        || context.dagCache.size == 0ull)
    {
        logErr()
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

    return true;
}
