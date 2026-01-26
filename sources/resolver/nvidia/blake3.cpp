#include <algo/hash_utils.hpp>
#include <algo/blake3/cuda/blake3.cuh>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <resolver/nvidia/blake3.hpp>


resolver::ResolverNvidiaBlake3::~ResolverNvidiaBlake3()
{
    blake3FreeMemory(parameters);
}


bool resolver::ResolverNvidiaBlake3::updateMemory(
    [[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    if (false == blake3InitMemory(parameters))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverNvidiaBlake3::updateConstants(
    [[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    overrideOccupancy(128u, 8192u);

    ////////////////////////////////////////////////////////////////////////////
    parameters.hostNonce = jobInfo.nonce;
    parameters.hostToGroup = jobInfo.toGroup;
    parameters.hostFromGroup = jobInfo.fromGroup;
    algo::copyHash(parameters.hostBoundary, jobInfo.boundary);
    algo::copyHash(parameters.hostTargetBlob, jobInfo.targetBlob);
    algo::copyHash(parameters.hostHeaderBlob, jobInfo.headerBlob);

    ////////////////////////////////////////////////////////////////////////////
    if (false == blake3UpateConstants(parameters))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverNvidiaBlake3::executeSync(
    [[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    parameters.hostNonce = jobInfo.nonce;
    blake3Search(cuStream[currentIndexStream],
                 parameters,
                 currentIndexStream,
                 blocks,
                 threads);
    CUDA_ER(cudaStreamSynchronize(cuStream[currentIndexStream]));
    CUDA_ER(cudaGetLastError());

    ////////////////////////////////////////////////////////////////////////////
    if (true == parameters.resultCache->found)
    {
        uint32_t const count
        {
            common::max_limit(parameters.resultCache->count, algo::blake3::MAX_RESULT)
        };

        resultShare.found = true;
        resultShare.fromGroup = jobInfo.fromGroup;
        resultShare.toGroup = jobInfo.toGroup;
        resultShare.count = count;
        resultShare.jobId = jobInfo.jobIDStr;
        resultShare.extraNonceSize = jobInfo.extraNonceSize;

        for (uint32_t i { 0u }; i < count; ++i)
        {
            resultShare.nonces[i] = parameters.resultCache->nonces[i];
        }

        parameters.resultCache->found = false;
        parameters.resultCache->count = 0u;
    }

    return true;
}


bool resolver::ResolverNvidiaBlake3::executeAsync(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    CUDA_ER(cudaStreamSynchronize(cuStream[currentIndexStream]));
    CUDA_ER(cudaGetLastError());

    ////////////////////////////////////////////////////////////////////////////
    swapIndexStrean();
    parameters.hostNonce = jobInfo.nonce;
    blake3Search(cuStream[currentIndexStream],
                 parameters,
                 currentIndexStream,
                 blocks,
                 threads);

    ////////////////////////////////////////////////////////////////////////////
    swapIndexStrean();
    algo::blake3::Result* resultCache { &parameters.resultCache[currentIndexStream] };
    if (true == resultCache->found)
    {
        uint32_t const count
        {
            common::max_limit(resultCache->count, algo::blake3::MAX_RESULT)
        };

        resultShare.found = true;
        resultShare.fromGroup = jobInfo.fromGroup;
        resultShare.toGroup = jobInfo.toGroup;
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


void resolver::ResolverNvidiaBlake3::submit(
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

                std::string nonceStr { nonceHexa.str() };

                while (nonceStr.size() < 48)
                {
                    nonceStr += "0";
                }

                boost::json::object params{};
                params["jobId"] = resultShare.jobId;
                params["fromGroup"] = resultShare.fromGroup;
                params["toGroup"] = resultShare.toGroup;
                params["nonce"] = nonceStr;

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }
    }

    resultShare.count = 0u;
    resultShare.found = false;
}


void resolver::ResolverNvidiaBlake3::submit(
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

                boost::json::object params{};
                params["jobId"] = resultShare.jobId;
                params["fromGroup"] = resultShare.fromGroup;
                params["toGroup"] = resultShare.toGroup;
                params["nonce"] = nonceHexa.str().substr(resultShare.extraNonceSize);

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }
    }

    resultShare.count = 0u;
    resultShare.found = false;
}
