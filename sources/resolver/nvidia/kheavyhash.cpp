#include <sstream>

#include <algo/hash_utils.hpp>
#include <algo/kheavyhash/cuda/kheavyhash.cuh>
#include <algo/kheavyhash/matrix.hpp>
#include <algo/kheavyhash/types.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <resolver/nvidia/kheavyhash.hpp>


resolver::ResolverNvidiaKHeavyHash::ResolverNvidiaKHeavyHash() : resolver::ResolverNvidia()
{
    algorithm = algo::ALGORITHM::KHEAVYHASH;
}


resolver::ResolverNvidiaKHeavyHash::~ResolverNvidiaKHeavyHash()
{
    kheavyhashFreeMemory(parameters);
}


bool resolver::ResolverNvidiaKHeavyHash::updateMemory([[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    // No DAG: just allocate the fixed-size device buffers once.
    return kheavyhashInitMemory(parameters);
}


bool resolver::ResolverNvidiaKHeavyHash::updateConstants(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    overrideOccupancy(256u, 8192u);

    ////////////////////////////////////////////////////////////////////////////
    parameters.hostNonce = jobInfo.nonce;
    parameters.hostTimestamp = jobInfo.timestamp;
    algo::copyHash(parameters.hostHeader, jobInfo.headerHash);
    algo::copyHash(parameters.hostTarget, jobInfo.boundary);

    ////////////////////////////////////////////////////////////////////////////
    // Host-side matrix generation (xoshiro256++ + full-rank gate) from the pre-pow
    // header -- the CPU reference the kernel is gated bit-identical against.
    ::kheavyhash::Hash256 seed{};
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        seed[i] = jobInfo.headerHash.ubytes[i];
    }
    ::kheavyhash::Matrix const matrix{ ::kheavyhash::generateMatrix(seed) };
    for (uint32_t r{ 0u }; r < 64u; ++r)
    {
        for (uint32_t c{ 0u }; c < 64u; ++c)
        {
            parameters.hostMatrix[r * 64u + c] = matrix[r][c];
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    return kheavyhashUpdateConstants(parameters);
}


bool resolver::ResolverNvidiaKHeavyHash::executeSync(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    parameters.hostNonce = jobInfo.nonce;
    kheavyhashSearch(cuStream[currentIndexStream], parameters, currentIndexStream, blocks, threads);
    CUDA_ER(cudaStreamSynchronize(cuStream[currentIndexStream]));
    CUDA_ER(cudaGetLastError());

    ////////////////////////////////////////////////////////////////////////////
    algo::kheavyhash::Result* resultCache{ &parameters.resultCache[currentIndexStream] };
    if (true == resultCache->found)
    {
        uint32_t const count{ common::max_limit(resultCache->count, algo::kheavyhash::MAX_RESULT) };

        resultShare.found = true;
        resultShare.count = count;
        resultShare.extraNonceSize = 0u;
        resultShare.jobId.assign(jobInfo.jobIDStr);

        for (uint32_t i{ 0u }; i < count; ++i)
        {
            resultShare.nonces[i] = resultCache->nonces[i];
        }

        resultCache->found = false;
        resultCache->count = 0u;
    }

    return true;
}


bool resolver::ResolverNvidiaKHeavyHash::executeAsync(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    CUDA_ER(cudaStreamSynchronize(cuStream[currentIndexStream]));
    CUDA_ER(cudaGetLastError());

    ////////////////////////////////////////////////////////////////////////////
    swapIndexStream();
    parameters.hostNonce = jobInfo.nonce;
    kheavyhashSearch(cuStream[currentIndexStream], parameters, currentIndexStream, blocks, threads);

    ////////////////////////////////////////////////////////////////////////////
    swapIndexStream();
    algo::kheavyhash::Result* resultCache{ &parameters.resultCache[currentIndexStream] };
    if (true == resultCache->found)
    {
        uint32_t const count{ common::max_limit(resultCache->count, algo::kheavyhash::MAX_RESULT) };

        resultShare.found = true;
        resultShare.count = count;
        resultShare.extraNonceSize = 0u;
        resultShare.jobId.assign(jobInfo.jobIDStr);

        for (uint32_t i{ 0u }; i < count; ++i)
        {
            resultShare.nonces[i] = resultCache->nonces[i];
        }

        resultCache->found = false;
        resultCache->count = 0u;
    }

    ////////////////////////////////////////////////////////////////////////////
    swapIndexStream();

    return true;
}


void resolver::ResolverNvidiaKHeavyHash::submit(stratum::Stratum* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i{ 0u }; i < resultShare.count; ++i)
            {
                std::stringstream nonceHexa;
                nonceHexa << std::hex << resultShare.nonces[i];

                boost::json::array params{ resultShare.jobId, nonceHexa.str() };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }

        resultShare.count = 0u;
        resultShare.found = false;
    }
}


void resolver::ResolverNvidiaKHeavyHash::submit(stratum::StratumSmartMining* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i{ 0u }; i < resultShare.count; ++i)
            {
                std::stringstream nonceHexa;
                nonceHexa << std::hex << resultShare.nonces[i];

                boost::json::array params{ resultShare.jobId, nonceHexa.str() };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }

        resultShare.count = 0u;
        resultShare.found = false;
    }
}
