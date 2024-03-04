#include <algo/hash_utils.hpp>
#include <algo/bitwise.hpp>
#include <algo/autolykos/cuda/autolykos.cuh>
#include <common/cast.hpp>
#include <common/error/cuda_error.hpp>
#include <common/log/log.hpp>
#include <resolver/nvidia/autolykos_v2.hpp>


bool resolver::ResolverNvidiaAutolykosV2::updateMemory(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    parameters.hostPeriod = castU32(jobInfo.period);
    parameters.hostHeight = algo::be::U32(castU32(jobInfo.blockNumber));
    parameters.hostDagItemCount = castU32(jobInfo.period);

    ////////////////////////////////////////////////////////////////////////////
    if (false == autolykosv2InitMemory(parameters))
    {
        return false;
    }
    ////////////////////////////////////////////////////////////////////////////
    if (false == autolykosv2BuildDag(cuStream, parameters))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverNvidiaAutolykosV2::updateConstants(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    setThreads(64u);
    setBlocks(131072u);

    ////////////////////////////////////////////////////////////////////////////
    parameters.hostNonce = jobInfo.nonce;
    parameters.hostPeriod = castU32(jobInfo.period);
    parameters.hostHeight = algo::be::U32(castU32(jobInfo.blockNumber));
    parameters.hostDagItemCount = castU32(jobInfo.period);
    algo::copyHash(parameters.hostBoundary, jobInfo.boundary);
    algo::copyHash(parameters.hostHeader, jobInfo.headerHash);
    if (false == autolykosv2UpateConstants(parameters))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverNvidiaAutolykosV2::execute(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    parameters.hostNonce = jobInfo.nonce;
    if (false == autolykosv2Search(cuStream, blocks, threads, parameters))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == parameters.resultCache->found)
    {
        resultShare.found = true;
        resultShare.count = parameters.resultCache->count;
        resultShare.jobId = jobInfo.jobIDStr;
        resultShare.extraNonceSize = jobInfo.extraNonceSize;
        resultShare.extraNonce2Size = jobInfo.extraNonce2Size;

        for (uint32_t i { 0u }; i < parameters.resultCache->count; ++i)
        {
            resultShare.nonces[i] = parameters.resultCache->nonces[i];
        }

        parameters.resultCache->found = false;
        parameters.resultCache->count = 0u;
    }


    ////////////////////////////////////////////////////////////////////////////
    return true;
}


void resolver::ResolverNvidiaAutolykosV2::submit(
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
                    nonceHexa.str().substr(resultShare.extraNonceSize),
                    nonceHexa.str()
                };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }

        }
    }

    resultShare.count = 0u;
    resultShare.found = false;
}


void resolver::ResolverNvidiaAutolykosV2::submit(
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
                    nonceHexa.str().substr(resultShare.extraNonceSize),
                    nonceHexa.str()
                };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }

        }
    }

    resultShare.count = 0u;
    resultShare.found = false;
}
