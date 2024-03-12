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


void resolver::ResolverNvidiaEthash::updateContext(
    stratum::StratumJobInfo const& jobInfo)
{
    algo::ethash::initializeDagContext(context,
                                       jobInfo.epoch,
                                       algo::ethash::MAX_EPOCH_NUMBER,
                                       algo::ethash::DAG_COUNT_ITEMS_GROWTH,
                                       algo::ethash::DAG_COUNT_ITEMS_INIT);
}


bool resolver::ResolverNvidiaEthash::updateMemory(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    updateContext(jobInfo);

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
        if (false == ethashSearch(getCurrentStream(),
                                  &parameters.resultCache[getCurrentIndex()],
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
        algo::ethash::Result& resultCache { parameters.resultCache[getCurrentIndex()] };
        if (true == resultCache.found)
        {
            resultShare.found = true;
            resultShare.count = resultCache.count;
            resultShare.jobId = jobInfo.jobIDStr;
            resultShare.extraNonceSize = jobInfo.extraNonceSize;

            for (uint32_t i { 0u }; i < resultCache.count; ++i)
            {
                resultShare.nonces[i] = resultCache.nonces[i];
            }

            resultCache.found = false;
            resultCache.count = 0u;
        }

    }
    else
    {
        ////////////////////////////////////////////////////////////////////////
        CUDA_ER(cudaStreamSynchronize(getCurrentStream()));
        CUDA_ER(cudaGetLastError());

        ////////////////////////////////////////////////////////////////////////
        if (false == ethashSearch(getNextStream(),
                                  &parameters.resultCache[getNextIndex()],
                                  blocks,
                                  threads,
                                  jobInfo.nonce))
        {
            return false;
        }


        ////////////////////////////////////////////////////////////////////////
        algo::ethash::Result& resultCache { parameters.resultCache[getCurrentIndex()] };
        if (true == resultCache.found)
        {
            resultShare.found = true;
            resultShare.count = resultCache.count;
            resultShare.jobId = jobInfo.jobIDStr;
            resultShare.extraNonceSize = jobInfo.extraNonceSize;

            for (uint32_t i { 0u }; i < resultCache.count; ++i)
            {
                resultShare.nonces[i] = resultCache.nonces[i];
            }

            resultCache.found = false;
            resultCache.count = 0u;
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
