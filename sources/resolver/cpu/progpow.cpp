#include <algo/hash_utils.hpp>
#include <algo/ethash/ethash.hpp>
#include <common/log/log.hpp>
#include <common/cast.hpp>
#include <common/config.hpp>
#include <resolver/cpu/progpow.hpp>


resolver::ResolverCpuProgPOW::~ResolverCpuProgPOW()
{
    SAFE_DELETE_ARRAY(parameters.headerCache);
    SAFE_DELETE_ARRAY(parameters.lightCache);
    SAFE_DELETE_ARRAY(parameters.dagCache);
}


bool resolver::ResolverCpuProgPOW::updateContext(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    algo::ethash::ContextGenerator::instance().build
    (
        algo::ALGORITHM::PROGPOW,
        context,
        jobInfo.epoch,
        maxEpoch,
        dagCountItemsGrowth,
        dagCountItemsInit,
        lightCacheCountItemsGrowth,
        lightCacheCountItemsInit,
        true
    );

    if (   0ull == context.lightCache.numberItem
        || 0ull == context.lightCache.size
        || 0ull == context.dagCache.numberItem
        || 0ull == context.dagCache.size)
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

    uint64_t const totalMemoryNeeded{ context.dagCache.size + context.lightCache.size };
    if (   0ull != deviceMemoryAvailable
        && totalMemoryNeeded >= deviceMemoryAvailable)
    {
        resolverErr()
            << "Device have not memory size available."
            << " Needed " << totalMemoryNeeded
            << ", memory available " << deviceMemoryAvailable;
        return false;
    }

    return true;
}


bool resolver::ResolverCpuProgPOW::updateMemory(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    if (false == updateContext(jobInfo))
    { 
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    SAFE_DELETE_ARRAY(parameters.headerCache);
    SAFE_DELETE_ARRAY(parameters.lightCache);
    SAFE_DELETE_ARRAY(parameters.dagCache);
    SAFE_DELETE(parameters.resultCache);

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const lightCacheDivisor{ algo::LEN_HASH_512 / algo::LEN_HASH_512_WORD_32 };
    uint32_t const dagDivisor{ algo::LEN_HASH_1024 / algo::LEN_HASH_1024_WORD_32 };
    parameters.lightCache = NEW_ARRAY(uint32_t, context.lightCache.size / lightCacheDivisor);
    parameters.dagCache = NEW_ARRAY(uint32_t, context.dagCache.size / dagDivisor);
    parameters.resultCache = NEW(algo::progpow::Result);

    IS_NULL(parameters.lightCache);
    IS_NULL(parameters.dagCache);
    IS_NULL(parameters.resultCache);

    for (uint32_t i = 0; i < context.lightCache.numberItem; ++i)
    {
        for (uint32_t j = 0u; j < algo::LEN_HASH_512_WORD_32; ++j)
        {
            parameters.lightCache[(i * algo::LEN_HASH_512_WORD_32) + j] = context.lightCache.hash[i].word32[j];
        }
    }

    return true;
}


bool resolver::ResolverCpuProgPOW::updateConstants(stratum::StratumJobInfo const& jobInfo)
{
    return true;
}


bool resolver::ResolverCpuProgPOW::executeSync(stratum::StratumJobInfo const& jobInfo)
{
    return true;
}


bool resolver::ResolverCpuProgPOW::executeAsync(stratum::StratumJobInfo const& jobInfo)
{
    return true;
}


void resolver::ResolverCpuProgPOW::submit(stratum::Stratum* const stratum)
{
}


void resolver::ResolverCpuProgPOW::submit(stratum::StratumSmartMining* const stratum)
{
}
