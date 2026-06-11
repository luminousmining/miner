#include <cstdint>

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <stratum/autolykos_v2.hpp>
#include <stratum/stratum.hpp>


////////////////////////////////////////////////////////////////////////////////
// Regression coverage for the NiceHash Autolykos2 boundary fix.
//
// Bug: NiceHash autolykos sends NO mining.set_difficulty / mining.set_target --
// the per-job target arrives embedded in mining.notify (params[6], a big decimal
// boundary). onMiningNotify() parsed jobInfo.boundary (hash256) from it but never
// set jobInfo.boundaryU64. Only onMiningSetDifficulty() did, and NiceHash never
// calls it, so boundaryU64 stayed 0. Stratum::isValidJob() rejects any job with
// boundaryU64 == 0 ("Boundary U64 == 0ull"), so every job was silently dropped
// and the GPU never started hashing.
//
// The fix recomputes boundaryU64 from the notify boundary, mirroring the other
// stratums (ethash/progpow/blake3) whose notify handlers always do so.
//
// No network needed: updateJob() is overridden to a no-op so onMiningNotify()'s
// only observable effect is the mutated jobInfo.
////////////////////////////////////////////////////////////////////////////////


struct ProbeStratumAutolykosV2 final : public stratum::StratumAutolykosV2
{
    void updateJob() override
    {
        // Skip dispatch/callback; the test inspects jobInfo directly.
    }
};


// A NiceHash-style mining.notify with the target embedded in params[6] and no
// preceding set_difficulty. Layout matches StratumAutolykosV2::onMiningNotify:
// [jobID, blockNumber, headerHash, "", "", _, boundaryDecimal, "", cleanJob].
boost::json::object makeNiceHashNotify()
{
    boost::json::object root;
    root["method"] = "mining.notify";
    root["params"] = boost::json::array{
        "0000000073fd34ab",                                                      // 0: job id
        1803772,                                                                 // 1: block number
        "6f109ba5226d1e0814cdeec79f1231d1d48196b5979a6d816e3621a1ef47ad80",      // 2: header hash
        "",                                                                      // 3
        "",                                                                      // 4
        2,                                                                       // 5
        "107839786668602559178668060348078522694548577690162289924414440996863", // 6: boundary
        "",                                                                      // 7
        true                                                                     // 8: clean job
    };
    return root;
}


struct StratumAutolykosV2NotifyTest : public testing::Test
{
    StratumAutolykosV2NotifyTest() = default;
    ~StratumAutolykosV2NotifyTest() = default;
};


////////////////////////////////////////////////////////////////////////////////
// The NiceHash notify carries the target in params[6] and no set_difficulty
// arrives first; boundaryU64 must still be populated or isValidJob() drops the
// job and the GPU never hashes.
////////////////////////////////////////////////////////////////////////////////

TEST_F(StratumAutolykosV2NotifyTest, NotifyEmbeddedBoundaryPopulatesBoundaryU64)
{
    ProbeStratumAutolykosV2 stratum{};

    stratum.onMiningNotify(makeNiceHashNotify());

    EXPECT_FALSE(algo::isHashEmpty(stratum.jobInfo.boundary));
    EXPECT_NE(0ull, stratum.jobInfo.boundaryU64);
}


////////////////////////////////////////////////////////////////////////////////
// The per-job notify boundary is authoritative: boundaryU64 must be derived from
// the same hash256 the resolver consumes, not left over from a prior difficulty.
////////////////////////////////////////////////////////////////////////////////

TEST_F(StratumAutolykosV2NotifyTest, NotifyBoundaryU64MatchesNotifyBoundaryHash)
{
    ProbeStratumAutolykosV2 stratum{};

    stratum.onMiningNotify(makeNiceHashNotify());

    EXPECT_EQ(algo::toUINT64(stratum.jobInfo.boundary), stratum.jobInfo.boundaryU64);
}
