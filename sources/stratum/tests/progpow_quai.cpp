#include <cstdint>

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include <algo/progpow/progpow_quai.hpp>
#include <stratum/progpow.hpp>
#include <stratum/progpow_quai.hpp>
#include <stratum/stratum_type.hpp>


////////////////////////////////////////////////////////////////////////////////
// Regression coverage for the progpow-quai DAG-epoch derivation.
//
// progpow-quai must key its DAG epoch off the block number, not the pool seed
// hash. Quai's seed hash is keccak-iterated on a much shorter cadence than its
// DAG-size epoch, so matching the seed (findEpoch) returns a wildly inflated
// epoch (586 at the block below) and the DAG balloons past the device's max
// single allocation -> clCreateBuffer fails with CL_INVALID_BUFFER_SIZE. The
// official quai-gpu-miner derives the epoch from blockNumber / EPOCH_LENGTH
// (388800) -> 11, and ignores the seed hash.
//
// The other ProgPoW coins keep the seed-hash-first derivation FiroPoW needs
// (its EPOCH_LENGTH changed across the chain), so the base stratum must still
// recover the epoch from the seed hash. Both paths are exercised below; no
// network or GPU is required.
////////////////////////////////////////////////////////////////////////////////

namespace
{
    // A real HeroMiners Quai (Ethereum/1.0.0) mining.notify captured live at block
    // 4'401'536. params[2] is the network seed hash = keccak256^586(0).
    boost::json::object makeQuaiNotify()
    {
        boost::json::object root;
        root["method"] = "mining.notify";
        root["params"] = boost::json::array{
            "0",
            "9f5842286250856890922ee77cbc5be3b2f418547f9d748527c4aacc6424c852",
            "15ca2ed4ba3db22b6df8b5d1b891b74b1dff6f3961ec08e4606dcfc9721072c8",
            "00000005ba03f80cf23190f76455c5b50637575fe8f983eab6727314d04c63e2",
            true,
            4401536,
            "1b01fa91"
        };
        return root;
    }
}


TEST(StratumProgpowQuaiEpoch, derivesEpochFromBlockNumberNotSeedHash)
{
    stratum::StratumProgpowQuai stratum{};
    stratum.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V1;

    stratum.onMiningNotify(makeQuaiNotify());

    EXPECT_EQ(4401536ull, stratum.jobInfo.blockNumber);
    EXPECT_EQ(static_cast<int32_t>(4401536ull / algo::progpow_quai::EPOCH_LENGTH), stratum.jobInfo.epoch);
    EXPECT_EQ(11, stratum.jobInfo.epoch);
}


TEST(StratumProgpowQuaiEpoch, baseProgpowStillPrefersSeedHash)
{
    stratum::StratumProgPOW stratum{};
    stratum.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V1;

    stratum.onMiningNotify(makeQuaiNotify());

    EXPECT_EQ(586, stratum.jobInfo.epoch);
}
