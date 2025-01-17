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
    if (false == blake3Search(cuStream[currentIndexStream],
                              parameters,
                              blocks,
                              threads))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == parameters.resultCache->found)
    {
        uint32_t const count
        {
            MAX_LIMIT(parameters.resultCache->count, algo::blake3::MAX_RESULT)
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
    return executeSync(jobInfo);
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

                logInfo() << "extraNonceSize: " << resultShare.extraNonceSize;
                logInfo() << "nonceHexa: " << nonceStr;
                logInfo() << "subNonce: " << nonceStr.substr(resultShare.extraNonceSize);

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
