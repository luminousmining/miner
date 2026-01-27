#include <algo/hash_utils.hpp>
#include <stratum/job_info.hpp>


void stratum::StratumJobInfo::copy(
    stratum::StratumJobInfo const& other)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtx);

    ////////////////////////////////////////////////////////////////////////////
    epoch = other.epoch;

    ////////////////////////////////////////////////////////////////////////////
    algo::copyHash(jobID, other.jobID);
    algo::copyHash(headerHash, other.headerHash);
    algo::copyHash(coinb1, other.coinb1);
    algo::copyHash(coinb2, other.coinb2);
    algo::copyHash(seedHash, other.seedHash);
    algo::copyHash(boundary, other.boundary);
    for (uint32_t i { 0u }; i < 12u; ++i)
    {
        algo::copyHash(merkletree[i], other.merkletree[i]);
    }

    ////////////////////////////////////////////////////////////////////////////
    nonce = other.nonce;
    startNonce = other.startNonce;
    extraNonce = other.extraNonce;
    gapNonce = other.gapNonce;
    blockNumber = other.blockNumber;
    period = other.period;
    boundaryU64 = other.boundaryU64;
    targetBits = other.targetBits;
    cleanJob = other.cleanJob;
    jobIDStr.assign(other.jobIDStr);

    ////////////////////////////////////////////////////////////////////////////
    // ETHASH && PROGPOW
    extraNonceSize = other.extraNonceSize;
    extraNonce2Size = other.extraNonce2Size;

    ////////////////////////////////////////////////////////////////////////////
    // Blake 3
    algo::copyHash(headerBlob, other.headerBlob);
    algo::copyHash(targetBlob, other.targetBlob);
    fromGroup = other.fromGroup;
    toGroup = other.toGroup;
}
