#include <algo/hash_utils.hpp>
#include <stratum/job_info.hpp>


stratum::StratumJobInfo::StratumJobInfo(
    stratum::StratumJobInfo const& obj)
{
    copy(obj);
}


stratum::StratumJobInfo& stratum::StratumJobInfo::operator=(
    stratum::StratumJobInfo const& obj)
{
    copy(obj);
    return *this;
}


void stratum::StratumJobInfo::copy(
    stratum::StratumJobInfo const& obj)
{
    epoch = obj.epoch;

    algo::copyHash(jobID, obj.jobID);
    algo::copyHash(headerHash, obj.headerHash);
    algo::copyHash(coinb1, obj.coinb1);
    algo::copyHash(coinb2, obj.coinb2);
    algo::copyHash(seedHash, obj.seedHash);
    algo::copyHash(boundary, obj.boundary);
    for (uint32_t i { 0u }; i < 12u; ++i)
    {
        algo::copyHash(merkletree[i], obj.merkletree[i]);
    }

    nonce = obj.nonce;
    startNonce = obj.startNonce;
    extraNonce = obj.extraNonce;
    extraNonceSize = obj.extraNonceSize;
    gapNonce = obj.gapNonce;
    blockNumber = obj.blockNumber;
    period = obj.period;
    boundaryU64 = obj.boundaryU64;
    targetBits = obj.targetBits;
    cleanJob = obj.cleanJob;
    jobIDStr.assign(obj.jobIDStr);
}
