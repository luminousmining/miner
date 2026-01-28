#include <algo/hash_utils.hpp>
#include <algo/bitwise.hpp>
#include <algo/autolykos/autolykos.hpp>
#include <algo/autolykos/cuda/autolykos.cuh>
#include <common/cast.hpp>
#include <common/error/cuda_error.hpp>
#include <common/log/log.hpp>
#include <resolver/nvidia/autolykos_v2.hpp>


resolver::ResolverNvidiaAutolykosV2::~ResolverNvidiaAutolykosV2()
{
    autolykosv2FreeMemory(parameters);
}


bool resolver::ResolverNvidiaAutolykosV2::updateMemory(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    parameters.hostPeriod = castU32(jobInfo.period);
    parameters.hostHeight = algo::be::uint32(castU32(jobInfo.blockNumber));
    parameters.hostDagItemCount = castU32(jobInfo.period);

    ////////////////////////////////////////////////////////////////////////////
    uint64_t const totalMemoryNeeded
    {
          algo::LEN_HASH_256
        + (parameters.hostDagItemCount * algo::LEN_HASH_256)
        + (algo::autolykos_v2::NONCES_PER_ITER * algo::LEN_HASH_256)
        + sizeof(algo::autolykos_v2::Result)
    };
    if (   0ull != deviceMemoryAvailable
        && totalMemoryNeeded >= deviceMemoryAvailable)
    {
        resolverErr()
            << "Device have not memory size available."
            << " Needed " << totalMemoryNeeded << ", memory available " << deviceMemoryAvailable;
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == autolykosv2InitMemory(parameters))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == autolykosv2BuildDag(cuStream[currentIndexStream], parameters))
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
    setBlocks(algo::autolykos_v2::NONCES_PER_ITER / 64u);

    ////////////////////////////////////////////////////////////////////////////
    parameters.hostNonce = jobInfo.nonce;
    parameters.hostPeriod = castU32(jobInfo.period);
    parameters.hostHeight = algo::be::uint32(castU32(jobInfo.blockNumber));
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


bool resolver::ResolverNvidiaAutolykosV2::executeSync(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    parameters.hostNonce = jobInfo.nonce;
    if (false == autolykosv2Search(cuStream[currentIndexStream], blocks, threads, parameters))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == parameters.resultCache->found)
    {
        uint32_t const count
        {
            common::max_limit(parameters.resultCache->count, algo::autolykos_v2::MAX_RESULT)
        };

        uint32_t indexValidNonce{ 0u };
        for (uint32_t i { 0u }; i < count; ++i)
        {
            auto const nonce{ parameters.resultCache->nonces[i] };
            auto const isValid
            {
                algo::autolykos_v2::mhssamadani::isValidShare
                (
                    parameters.hostHeader,
                    parameters.hostBoundary,
                    nonce,
                    parameters.hostHeight
                )
            };
            resolverDebug() << "test nonce[" << std::hex << nonce << "] is " << std::boolalpha << isValid;
            if (true == isValid)
            {
                resultShare.found = true;
                resultShare.nonces[indexValidNonce] = nonce;
                ++indexValidNonce;
            }
        }

        if (true == resultShare.found)
        {
            resultShare.count = indexValidNonce;
            resultShare.jobId = jobInfo.jobIDStr;
            resultShare.extraNonceSize = jobInfo.extraNonceSize;
            resultShare.extraNonce2Size = jobInfo.extraNonce2Size;
        }

        parameters.resultCache->found = false;
        parameters.resultCache->count = 0u;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverNvidiaAutolykosV2::executeAsync(
    stratum::StratumJobInfo const& jobInfo)
{
    return executeSync(jobInfo);
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

        resultShare.count = 0u;
        resultShare.found = false;
    }
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

        resultShare.count = 0u;
        resultShare.found = false;
    }
}
