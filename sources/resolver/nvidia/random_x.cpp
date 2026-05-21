#if defined(CUDA_ENABLE)

#include <iomanip>
#include <sstream>

#include <algo/hash_utils.hpp>
#include <algo/random_x/cuda/random_x.cuh>
#include <common/custom.hpp>
#include <common/error/cuda_error.hpp>
#include <common/log/log.hpp>
#include <resolver/nvidia/random_x.hpp>


resolver::ResolverNvidiaRandomX::ResolverNvidiaRandomX() : resolver::ResolverNvidia()
{
    algorithm = algo::ALGORITHM::RANDOM_X;
}


resolver::ResolverNvidiaRandomX::~ResolverNvidiaRandomX()
{
    randomxFreeMemory(parameters);
}


bool resolver::ResolverNvidiaRandomX::updateMemory(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // Only rebuild dataset when the seed hash changes
    if (true == algo::isEqual(jobInfo.seedHash, parameters.hostSeedHash)
        && nullptr != parameters.dataset)
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    resolverErr() << "RandomX: rebuilding dataset (seed changed)";

    ////////////////////////////////////////////////////////////////////////////
    randomxFreeMemory(parameters);

    if (false == randomxInitMemory(parameters, blocks, threads))
    {
        resolverErr() << "RandomX: failed to allocate GPU memory";
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Allocate temporary GPU cache (256 MiB), build via CPU Argon2d, then build dataset
    constexpr uint64_t CACHE_BYTES{ 268435456ull }; // 256 MiB
    uint8_t* cache{ nullptr };
    CU_ALLOC(&cache, CACHE_BYTES);

    if (false == randomxBuildCache(
            cuStream[currentIndexStream],
            cache,
            jobInfo.seedHash.ubytes))
    {
        CU_SAFE_DELETE(cache);
        resolverErr() << "RandomX: failed to build cache";
        return false;
    }

    if (false == randomxBuildDataset(
            cuStream[currentIndexStream],
            parameters,
            cache,
            jobInfo.seedHash.ubytes))
    {
        CU_SAFE_DELETE(cache);
        resolverErr() << "RandomX: failed to build dataset";
        return false;
    }

    CU_SAFE_DELETE(cache);

    ////////////////////////////////////////////////////////////////////////////
    algo::copyHash(parameters.hostSeedHash, jobInfo.seedHash);

    return true;
}


bool resolver::ResolverNvidiaRandomX::updateConstants(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // headerBlob.ubytes[0..76] holds the 77-byte Monero block template
    parameters.hostNonce  = jobInfo.nonce;
    parameters.hostTarget = jobInfo.targetBits;

    if (false == randomxUpdateConstants(
            jobInfo.headerBlob.ubytes,
            jobInfo.targetBits,
            jobInfo.nonce))
    {
        return false;
    }

    return true;
}


bool resolver::ResolverNvidiaRandomX::executeSync(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    parameters.hostNonce = jobInfo.nonce;

    if (false == randomxSearch(cuStream[currentIndexStream], blocks, threads, parameters))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == parameters.resultCache->found)
    {
        resultShare.found    = true;
        resultShare.count    = 1u;
        resultShare.jobId    = jobInfo.jobIDStr;
        resultShare.nonces[0] = parameters.resultCache->nonces[0];

        for (uint32_t b{ 0u }; b < 32u; ++b)
        {
            resultShare.hashes[0][b] = parameters.resultCache->hash[b];
        }

        parameters.resultCache->found = false;
        parameters.resultCache->count = 0u;
    }

    return true;
}


bool resolver::ResolverNvidiaRandomX::executeAsync(stratum::StratumJobInfo const& jobInfo)
{
    return executeSync(jobInfo);
}


void resolver::ResolverNvidiaRandomX::submit(stratum::Stratum* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i{ 0u }; i < resultShare.count; ++i)
            {
                // Nonce: 8 hex chars, uint32 LE encoding
                std::ostringstream nonceStream;
                nonceStream << std::hex << std::setfill('0') << std::setw(8)
                            << resultShare.nonces[i];

                // Hash: 64 hex chars, raw 32 bytes
                std::ostringstream hashStream;
                for (uint32_t b{ 0u }; b < 32u; ++b)
                {
                    hashStream << std::hex << std::setfill('0') << std::setw(2)
                               << static_cast<uint32_t>(resultShare.hashes[i][b]);
                }

                boost::json::array params
                {
                    resultShare.jobId,
                    nonceStream.str(),
                    hashStream.str()
                };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0u;
            }
        }

        resultShare.count = 0u;
        resultShare.found = false;
    }
}


void resolver::ResolverNvidiaRandomX::submit(stratum::StratumSmartMining* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i{ 0u }; i < resultShare.count; ++i)
            {
                std::ostringstream nonceStream;
                nonceStream << std::hex << std::setfill('0') << std::setw(8)
                            << resultShare.nonces[i];

                std::ostringstream hashStream;
                for (uint32_t b{ 0u }; b < 32u; ++b)
                {
                    hashStream << std::hex << std::setfill('0') << std::setw(2)
                               << static_cast<uint32_t>(resultShare.hashes[i][b]);
                }

                boost::json::array params
                {
                    resultShare.jobId,
                    nonceStream.str(),
                    hashStream.str()
                };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0u;
            }
        }

        resultShare.count = 0u;
        resultShare.found = false;
    }
}

#endif
