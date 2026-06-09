#include <algo/ethash/ethash.hpp>
#include <algo/progpow/firopow.hpp>
#include <common/cast.hpp>
#include <stratum/firopow.hpp>


stratum::StratumFiroPOW::StratumFiroPOW() : stratum::StratumProgPOW()
{
    maxPeriod = algo::firopow::MAX_PERIOD;
    maxEpochLength = algo::firopow::EPOCH_LENGTH;
}


int32_t stratum::StratumFiroPOW::deriveEpoch(stratum::StratumJobInfo const& jobInfo) const
{
    ////////////////////////////////////////////////////////////////////////////
    // Firo's FiroPoW epoch length changed across the chain's history, so the base
    // blockNumber/EPOCH_LENGTH mapping derives the wrong epoch (wrong DAG ->
    // "Invalid Mixhash" rejects) on live Firo. The network seed hash encodes the
    // authoritative epoch, so prefer it; fall back to blockNumber/EPOCH_LENGTH only
    // when the seed isn't recognized.
    int32_t const epoch{ algo::ethash::ContextGenerator::instance().findEpoch(jobInfo.seedHash, maxEthashEpoch) };
    if (-1 == epoch && jobInfo.blockNumber > 0ull)
    {
        return cast32(jobInfo.blockNumber / castU64(maxEpochLength));
    }
    return epoch;
}
