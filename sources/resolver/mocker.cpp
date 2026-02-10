#if defined(TOOL_MOCKER)

#include <boost/chrono/duration.hpp>


#include <common/log/log.hpp>
#include <common/config.hpp>
#include <resolver/mocker.hpp>

#include <algo/ethash/ethash.hpp>
#include <algo/progpow/progpow.hpp>


static constexpr boost::chrono::milliseconds WAIT_UPDATE_MEMORY{ 40000 };
static constexpr boost::chrono::milliseconds WAIT_UPDATE_CONSTANT{ 2000 };
static constexpr boost::chrono::milliseconds WAIT_UPDATE_EXECUTE_SYNC{ 1000 };
static constexpr boost::chrono::milliseconds WAIT_UPDATE_EXECUTE_ASYNC{ 1000 };


void  resolver::ResolverMocker::overrideOccupancy(
    [[maybe_unused]] uint32_t const defaultThreads,
    [[maybe_unused]] uint32_t const defaultBlocks)
{
}


bool resolver::ResolverMocker::updateMemory(
    [[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    ///////////////////////////////////////////////////////////////////////////
    common::Config& config{ common::Config::instance() };

    ///////////////////////////////////////////////////////////////////////////
    algo::DagContext context{};
    uint32_t maxEpoch{ algo::ethash::MAX_EPOCH_NUMBER };
    uint32_t lightCacheCountItemsGrowth{ algo::ethash::LIGHT_CACHE_COUNT_ITEMS_GROWTH };
    uint32_t lightCacheCountItemsInit{ algo::ethash::LIGHT_CACHE_COUNT_ITEMS_INIT };
    uint32_t dagItemParents{ algo::ethash::DAG_ITEM_PARENTS };
    uint32_t dagCountItemsGrowth{ algo::ethash::DAG_COUNT_ITEMS_GROWTH };
    uint32_t dagCountItemsInit{ algo::ethash::DAG_COUNT_ITEMS_INIT };
    uint32_t countCache{ algo::progpow::v_0_9_3::COUNT_CACHE };
    uint32_t countMath{ algo::progpow::v_0_9_3::COUNT_MATH };
    algo::ethash::initializeDagContext
    (
        context,
        jobInfo.epoch,
        maxEpoch,
        dagCountItemsGrowth,
        dagCountItemsInit,
        lightCacheCountItemsGrowth,
        lightCacheCountItemsInit
    );
    algo::ethash::buildLightCache(context, config.deviceAlgorithm.ethashBuildLightCacheCPU);

    ///////////////////////////////////////////////////////////////////////////
    boost::this_thread::sleep_for(WAIT_UPDATE_MEMORY);

    ///////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverMocker::updateConstants(
    [[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    boost::this_thread::sleep_for(WAIT_UPDATE_CONSTANT);
    return true;
}


bool resolver::ResolverMocker::executeSync(
    [[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    boost::this_thread::sleep_for(WAIT_UPDATE_EXECUTE_SYNC);
    return true;
}


bool resolver::ResolverMocker::executeAsync(
    [[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    boost::this_thread::sleep_for(WAIT_UPDATE_EXECUTE_ASYNC);
    return true;
}


void resolver::ResolverMocker::submit(
    [[maybe_unused]] stratum::Stratum* const stratum)
{
}


void resolver::ResolverMocker::submit(
    [[maybe_unused]] stratum::StratumSmartMining* const stratum)
{
}

#endif
