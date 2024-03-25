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
    if (false == ethashBuildDag(getCurrentStream(),
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

    setBlocks(8192u);
    setThreads(128u);

    return true;
}


bool resolver::ResolverNvidiaEthash::execute(
    stratum::StratumJobInfo const& jobInfo)
{
    if (false == isDoubleStream)
    {
        ////////////////////////////////////////////////////////////////////////
        algo::ethash::Result& paramResultCache { parameters.resultCache[getCurrentIndex()] };
        if (false == ethashSearch(getCurrentStream(),
                                  &paramResultCache,
                                  blocks,
                                  threads,
                                  jobInfo.nonce))
        {
            return false;
        }

        ////////////////////////////////////////////////////////////////////////
        CUDA_ER(cudaStreamSynchronize(getCurrentStream()));
        CUDA_ER(cudaGetLastError());

        ////////////////////////////////////////////////////////////////////////
        if (true == paramResultCache.found)
        {
            uint32_t const count { MAX_LIMIT(paramResultCache.count, 4u) };

            resultShare.found = true;
            resultShare.count = count;
            resultShare.jobId = jobInfo.jobIDStr;
            resultShare.extraNonceSize = jobInfo.extraNonceSize;

            for (uint32_t i { 0u }; i < count; ++i)
            {
                resultShare.nonces[i] = paramResultCache.nonces[i];
            }

            paramResultCache.found = false;
            paramResultCache.count = 0u;
        }
    }
    else
    {
        ////////////////////////////////////////////////////////////////////////
        CUDA_ER(cudaStreamSynchronize(getCurrentStream()));
        CUDA_ER(cudaGetLastError());

        ////////////////////////////////////////////////////////////////////////
        algo::ethash::Result& paramResultCache { parameters.resultCache[getCurrentIndex()] };
        if (false == ethashSearch(getNextStream(),
                                  &parameters.resultCache[getNextIndex()],
                                  blocks,
                                  threads,
                                  jobInfo.nonce))
        {
            return false;
        }

        ////////////////////////////////////////////////////////////////////////
        if (true == paramResultCache.found)
        {
            uint32_t const count { MAX_LIMIT(paramResultCache.count, 4u) };

            resultShare.found = true;
            resultShare.count = count;
            resultShare.jobId = jobInfo.jobIDStr;
            resultShare.extraNonceSize = jobInfo.extraNonceSize;

            for (uint32_t i { 0u }; i < count; ++i)
            {
                resultShare.nonces[i] = paramResultCache.nonces[i];
            }

            paramResultCache.found = false;
            paramResultCache.count = 0u;
        }

        ////////////////////////////////////////////////////////////////////////
        swapStream();
    }

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
