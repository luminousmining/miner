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
    if (false == ethashBuildDag(cuStream,
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
    if (false == ethashSearch(cuStream,
                              parameters.resultCache,
                              blocks,
                              threads,
                              jobInfo.nonce))
    {
        return false;
    }

    if (true == parameters.resultCache->found)
    {
        resultShare.found = true;
        resultShare.count = parameters.resultCache->count;
        resultShare.jobId = jobInfo.jobIDStr;
        resultShare.extraNonceSize = jobInfo.extraNonceSize;

        for (uint32_t i { 0u }; i < parameters.resultCache->count; ++i)
        {
            resultShare.nonces[i] = parameters.resultCache->nonces[i];
        }

        parameters.resultCache->found = false;
        parameters.resultCache->count = 0u;
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
