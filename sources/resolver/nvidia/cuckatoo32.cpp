#include <cstring>

#include <algo/cuckatoo/blake2b_256.hpp>
#include <algo/cuckatoo/cuda/cuckatoo32.cuh>
#include <algo/cuckatoo/cuckatoo.hpp>
#include <algo/cuckatoo/result.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <resolver/nvidia/cuckatoo32.hpp>
#include <stratum/smart_mining.hpp>
#include <stratum/stratum.hpp>

#if defined(CUDA_ENABLE)


resolver::ResolverNvidiaCuckatoo32::ResolverNvidiaCuckatoo32()
{
    // blocks=1, threads=1 → batchNonce=1×1=1 (each kernel call = one complete graph)
    // The actual kernel grid (DEFAULT_BLOCKS × DEFAULT_THREADS) is hardcoded in cuckatoo32.cu.
    overrideOccupancy(algo::cuckatoo::DEFAULT_THREADS, algo::cuckatoo::DEFAULT_BLOCKS);
}


resolver::ResolverNvidiaCuckatoo32::~ResolverNvidiaCuckatoo32()
{
    cuckatoo32FreeMemory(parameters);
}


bool resolver::ResolverNvidiaCuckatoo32::updateMemory(
    [[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    return cuckatoo32AllocMemory(parameters);
}


bool resolver::ResolverNvidiaCuckatoo32::updateConstants(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // Store job context for submit()
    resultShare.jobId     = jobInfo.jobIDStr;
    resultShare.height    = jobInfo.blockNumber;
    resultShare.grinJobId = jobInfo.grinJobId;
    resultShare.nonce     = jobInfo.nonce;
    resultShare.found     = false;

    parameters.nonce = jobInfo.nonce;

    ////////////////////////////////////////////////////////////////////////////
    // Derive SipHash keys:
    //   input  = pre_pow_bytes (jobInfo.prePowSize bytes from headerBlob)
    //            || nonce encoded as 8 bytes little-endian
    //   H1     = blake2b-256(input)
    //   H2     = blake2b-256(H1)
    //   k0..k3 = H2 as four little-endian uint64_t
    ////////////////////////////////////////////////////////////////////////////
    uint32_t const prePowSize{ jobInfo.prePowSize };
    uint8_t const* prePow{ jobInfo.headerBlob.ubytes };

    // Build input buffer: pre_pow || nonce_le8
    std::vector<uint8_t> input(prePowSize + 8u);
    std::memcpy(input.data(), prePow, prePowSize);
    uint64_t const nonce{ jobInfo.nonce };
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        input[prePowSize + i] = static_cast<uint8_t>((nonce >> (8u * i)) & 0xFFu);
    }

    // H1 = blake2b-256(pre_pow || nonce_le8)
    uint8_t h1[32]{};
    algo::cuckatoo::blake2b256(input.data(), input.size(), h1);

    // H2 = blake2b-256(H1)
    uint8_t h2[32]{};
    algo::cuckatoo::blake2b256(h1, 32u, h2);

    // Extract keys (little-endian uint64)
    auto le64 = [](uint8_t const* p) -> uint64_t
    {
        return static_cast<uint64_t>(p[0])
             | (static_cast<uint64_t>(p[1]) <<  8u)
             | (static_cast<uint64_t>(p[2]) << 16u)
             | (static_cast<uint64_t>(p[3]) << 24u)
             | (static_cast<uint64_t>(p[4]) << 32u)
             | (static_cast<uint64_t>(p[5]) << 40u)
             | (static_cast<uint64_t>(p[6]) << 48u)
             | (static_cast<uint64_t>(p[7]) << 56u);
    };

    parameters.k0 = le64(h2 +  0u);
    parameters.k1 = le64(h2 +  8u);
    parameters.k2 = le64(h2 + 16u);
    parameters.k3 = le64(h2 + 24u);

    ////////////////////////////////////////////////////////////////////////////
    // Upload keys to GPU constant memory
    return cuckatoo32UpdateConstants(parameters);
}


bool resolver::ResolverNvidiaCuckatoo32::executeSync(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // Update nonce so the result can be submitted
    resultShare.nonce = jobInfo.nonce;
    parameters.nonce  = jobInfo.nonce;

    __TRACE();
    ////////////////////////////////////////////////////////////////////////////
    if (false == cuckatoo32Trim(cuStream[0], parameters))
    {
    __TRACE();
        return false;
    }

    __TRACE();
    ////////////////////////////////////////////////////////////////////////////
    bool     found{ false };
    uint32_t proof[algo::cuckatoo::PROOF_SIZE]{};
    if (false == cuckatoo32FindCycle(parameters, &found, proof))
    {
    __TRACE();
        return false;
    }

    __TRACE();
    ////////////////////////////////////////////////////////////////////////////
    if (true == found)
    {
        resultShare.found = true;
        std::copy(proof, proof + algo::cuckatoo::PROOF_SIZE, resultShare.proof);
    }

    __TRACE();
    logInfo() << "executed";

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverNvidiaCuckatoo32::executeAsync(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // Cuckatoo32 uses nearly all available VRAM, making true double-buffering
    // impractical on most GPUs.  Fall back to synchronous execution.
    return executeSync(jobInfo);
}


void resolver::ResolverNvidiaCuckatoo32::submit(stratum::Stratum* const stratum)
{
    if (false == resultShare.found || nullptr == stratum) { return; }

    boost::json::array powArray;
    for (uint32_t i{ 0u }; i < algo::cuckatoo::PROOF_SIZE; ++i)
    {
        powArray.push_back(resultShare.proof[i]);
    }

    boost::json::object params;
    params["height"]  = resultShare.height;
    params["job_id"]  = resultShare.grinJobId;
    params["nonce"]   = resultShare.nonce;
    params["pow"]     = powArray;

    stratum->miningSubmit(deviceId, params);

    resultShare.found = false;
}


void resolver::ResolverNvidiaCuckatoo32::submit(
    [[maybe_unused]] stratum::StratumSmartMining* const stratum)
{
    // Cuckatoo32 is not supported by the Smart Mining profile.
}

#endif
