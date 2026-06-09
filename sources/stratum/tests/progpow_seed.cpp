#include <cstdint>

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include <algo/progpow/firopow.hpp>
#include <algo/progpow/progpow_quai.hpp>
#include <stratum/firopow.hpp>
#include <stratum/progpow_quai.hpp>
#include <stratum/stratum_type.hpp>


////////////////////////////////////////////////////////////////////////////////
// Regression coverage for the ProgPoW DAG-epoch derivation.
//
// progpow-quai keys its DAG epoch off the block number, not the pool seed hash.
// Quai's seed hash is keccak-iterated on a much shorter cadence than its DAG-size
// epoch, so matching the seed (findEpoch) returns a wildly inflated epoch (586 at
// the block below) and the DAG balloons past the device's max single allocation
// -> clCreateBuffer fails with CL_INVALID_BUFFER_SIZE. The official quai-gpu-miner
// derives the epoch from blockNumber / EPOCH_LENGTH (388800) -> 11. That is the
// base StratumProgPOW behaviour, shared by kawpow/meowpow/evrprogpow.
//
// FiroPoW is the exception: its EPOCH_LENGTH changed across the chain's history,
// so StratumFiroPOW overrides deriveEpoch to recover the authoritative epoch from
// the seed hash. Both paths are exercised below; no network or GPU is required.
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


    // A real WoolyPooly Firo (Ethereum/1.0.0) mining.notify. params[2] is the
    // network seed hash for FiroPoW epoch 650. Firo changed its epoch length across
    // the chain's history, so blockNumber / EPOCH_LENGTH (1'319'805 / 1300 = 1015)
    // derives the WRONG epoch -> wrong DAG -> the live "Invalid Firo Mixhash" reject.
    // StratumFiroPOW recovers the authoritative 650 from the seed hash instead.
    boost::json::object makeFiroNotify()
    {
        boost::json::object root;
        root["method"] = "mining.notify";
        root["params"] = boost::json::array{
            "0",
            "9f5842286250856890922ee77cbc5be3b2f418547f9d748527c4aacc6424c852",
            "969685223d756d0d2c314efcb880b13fd979b38e23cfc77bbf3d66e69949566e",
            "00000005ba03f80cf23190f76455c5b50637575fe8f983eab6727314d04c63e2",
            true,
            1319805,
            "1b01fa91"
        };
        return root;
    }
}


TEST(StratumProgpowSeed, quaiEpoch)
{
    stratum::StratumProgpowQuai stratum{};
    stratum.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V1;

    stratum.onMiningNotify(makeQuaiNotify());

    EXPECT_EQ(4401536ull, stratum.jobInfo.blockNumber);
    EXPECT_EQ(static_cast<int32_t>(4401536ull / algo::progpow_quai::EPOCH_LENGTH), stratum.jobInfo.epoch);
    EXPECT_EQ(11, stratum.jobInfo.epoch);
}


TEST(StratumProgpowSeed, firopowEpoch)
{
    stratum::StratumFiroPOW stratum{};
    stratum.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V1;

    stratum.onMiningNotify(makeFiroNotify());

    // FiroPoW recovers the authoritative epoch the seed hash encodes (650), not the
    // wrong blockNumber / EPOCH_LENGTH value (1'319'805 / 1300 = 1015) that caused
    // the live "Invalid Firo Mixhash" reject.
    EXPECT_EQ(650, stratum.jobInfo.epoch);
    EXPECT_NE(static_cast<int32_t>(1319805ull / algo::firopow::EPOCH_LENGTH), stratum.jobInfo.epoch);
}
