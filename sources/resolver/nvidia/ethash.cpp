#include <algo/keccak.hpp>
#include <algo/ethash/ethash.hpp>
#include <algo/ethash/cuda/ethash.cuh>
#include <common/cast.hpp>
#include <common/chrono.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <resolver/nvidia/ethash.hpp>


resolver::ResolverNvidiaEthash::~ResolverNvidiaEthash()
{
    ethashFreeMemory(parameters);
}


bool resolver::ResolverNvidiaEthash::updateContext(
    stratum::StratumJobInfo const& jobInfo)
{
    algo::ethash::initializeDagContext(context,
                                       jobInfo.epoch,
                                       algo::ethash::MAX_EPOCH_NUMBER,
                                       dagCountItemsGrowth,
                                       dagCountItemsInit,
                                       lightCacheCountItemsGrowth,
                                       lightCacheCountItemsInit);

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
    if (   0ull < deviceMemoryAvailable
        && totalMemoryNeeded >= deviceMemoryAvailable)
    {
        resolverErr()
            << "Device have not memory size available."
            << " Needed " << totalMemoryNeeded << ", memory available " << deviceMemoryAvailable;
        return false;
    }

    return true;
}


bool resolver::ResolverNvidiaEthash::updateMemory(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    if (false == updateContext(jobInfo))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == ethashInitMemory(context, parameters))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == ethashBuildDag(cuStream[currentIndexStream],
                                algo::ethash::DAG_ITEM_PARENTS,
                                castU32(context.dagCache.numberItem)))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverNvidiaEthash::updateConstants(
    stratum::StratumJobInfo const& jobInfo)
{
    uint32_t const* const header { jobInfo.headerHash.word32 };
    uint64_t const boundary { jobInfo.boundaryU64 };
    if (false == ethashUpdateConstants(header, boundary))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    overrideOccupancy(128u, 8192u);

    return true;
}


bool resolver::ResolverNvidiaEthash::executeSync(
    stratum::StratumJobInfo const& jobInfo)
{
    ethashSearch(cuStream[currentIndexStream],
                 &parameters.resultCache[currentIndexStream],
                 blocks,
                 threads,
                 jobInfo.nonce);
    CUDA_ER(cudaStreamSynchronize(cuStream[currentIndexStream]));
    CUDA_ER(cudaGetLastError());

    algo::ethash::Result* resultCache{ &parameters.resultCache[currentIndexStream] };
    if (true == resultCache->found)
    {
        uint32_t const count
        {
            MAX_LIMIT(resultCache->count, algo::ethash::MAX_RESULT)
        };

        resultShare.found = true;
        resultShare.count = count;
        resultShare.jobId = jobInfo.jobIDStr;
        resultShare.extraNonceSize = jobInfo.extraNonceSize;

        for (uint32_t i { 0u }; i < count; ++i)
        {
            resultShare.nonces[i] = resultCache->nonces[i];
        }

        resultCache->found = false;
        resultCache->count = 0u;
    }

    return true;
}


bool resolver::ResolverNvidiaEthash::executeAsync(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    CUDA_ER(cudaStreamSynchronize(cuStream[currentIndexStream]));
    CUDA_ER(cudaGetLastError());

    ////////////////////////////////////////////////////////////////////////////
    swapIndexStrean();
    ethashSearch(cuStream[currentIndexStream],
                 &parameters.resultCache[currentIndexStream],
                 blocks,
                 threads,
                 jobInfo.nonce);

    ////////////////////////////////////////////////////////////////////////////
    swapIndexStrean();
    algo::ethash::Result* resultCache{ &parameters.resultCache[currentIndexStream] };
    if (true == resultCache->found)
    {
        uint32_t const count
        {
            MAX_LIMIT(resultCache->count, algo::ethash::MAX_RESULT)
        };

        resultShare.found = true;
        resultShare.count = count;
        resultShare.jobId = jobInfo.jobIDStr;
        resultShare.extraNonceSize = jobInfo.extraNonceSize;

        for (uint32_t i { 0u }; i < count; ++i)
        {
            resultShare.nonces[i] = resultCache->nonces[i];
        }

        resultCache->found = false;
        resultCache->count = 0u;
    }

    ////////////////////////////////////////////////////////////////////////////
    swapIndexStrean();

    return true;
}


void resolver::ResolverNvidiaEthash::submit(
    stratum::Stratum* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i { 0u }; i < resultShare.count; ++i)
            {
                std::stringstream nonceHexa;
                nonceHexa << std::hex << resultShare.nonces[i];

                boost::json::array params
                {
                    resultShare.jobId,
                    nonceHexa.str().substr(resultShare.extraNonceSize)
                };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }

        resultShare.count = 0u;
        resultShare.found = false;
        resultShare.extraNonceSize = 0;
    }
}


void resolver::ResolverNvidiaEthash::submit(
    stratum::StratumSmartMining* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i { 0u }; i < resultShare.count; ++i)
            {
                std::stringstream nonceHexa;
                nonceHexa << std::hex << resultShare.nonces[i];

                boost::json::array params
                {
                    resultShare.jobId,
                    nonceHexa.str().substr(resultShare.extraNonceSize)
                };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }

        resultShare.count = 0u;
        resultShare.found = false;
        resultShare.extraNonceSize = 0;
    }
}
