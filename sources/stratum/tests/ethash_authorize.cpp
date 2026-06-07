#include <cstdint>
#include <utility>

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include <stratum/ethash.hpp>
#include <stratum/stratum.hpp>
#include <stratum/stratum_type.hpp>


////////////////////////////////////////////////////////////////////////////////
// Regression coverage for the NiceHash subscribe/authorize fix (PR #133).
//
// Bug: miningAuthorize() used to be reached only after result.at(1) parsed
// successfully, so on EthereumStratum/1.0.0 (NiceHash) any missing/short/oddly
// typed extranonce left the worker unauthenticated and the pool rejected every
// share ("Missing KYC/KYB"). The fix authorizes on every non-error subscribe
// reply for ETHEREUM_V1, while keeping the original ordering for other types.
//
// These tests need no network: miningAuthorize() is the observable side effect
// of a successful subscribe, so we override it to count calls instead of
// sending. onResponse()'s subscribe path otherwise only mutates plain state.
////////////////////////////////////////////////////////////////////////////////


namespace
{
    struct ProbeStratumEthash final : public stratum::StratumEthash
    {
        uint32_t authorizeCount{ 0u };

        void miningAuthorize() override
        {
            ++authorizeCount;
        }
    };


    boost::json::object makeSubscribeReply(boost::json::value result)
    {
        boost::json::object root;
        root["id"] = stratum::Stratum::ID_MINING_SUBSCRIBE;
        root["error"] = nullptr;
        root["result"] = std::move(result);
        return root;
    }
}


struct StratumEthashAuthorizeTest : public testing::Test
{
    StratumEthashAuthorizeTest() = default;
    ~StratumEthashAuthorizeTest() override = default;
};


////////////////////////////////////////////////////////////////////////////////
// ETHEREUM_V1 (NiceHash): authorize on ANY non-error subscribe reply.
////////////////////////////////////////////////////////////////////////////////

TEST_F(StratumEthashAuthorizeTest, V1ValidResultParsesNonceAndAuthorizes)
{
    ProbeStratumEthash stratum{};
    stratum.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V1;

    boost::json::array const notify{ "mining.notify", "deadbeef", "EthereumStratum/1.0.0" };
    stratum.onResponse(makeSubscribeReply(boost::json::array{ notify, "a1b2" }));

    EXPECT_EQ(1u, stratum.authorizeCount);
    EXPECT_EQ(0xa1b2u, stratum.jobInfo.targetBits);
}


TEST_F(StratumEthashAuthorizeTest, V1MissingResultStillAuthorizes)
{
    ProbeStratumEthash stratum{};
    stratum.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V1;

    // No "result" key at all -- the old code threw / skipped authorize here.
    boost::json::object root;
    root["id"] = stratum::Stratum::ID_MINING_SUBSCRIBE;
    root["error"] = nullptr;
    stratum.onResponse(root);

    EXPECT_EQ(1u, stratum.authorizeCount);
}


TEST_F(StratumEthashAuthorizeTest, V1ShortResultStillAuthorizes)
{
    ProbeStratumEthash stratum{};
    stratum.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V1;

    // result has only the notify sub-array, no extranonce element [1].
    boost::json::array const result{ boost::json::array{ "mining.notify", "deadbeef" } };
    stratum.onResponse(makeSubscribeReply(result));

    EXPECT_EQ(1u, stratum.authorizeCount);
}


TEST_F(StratumEthashAuthorizeTest, V1NonStringNonceStillAuthorizes)
{
    ProbeStratumEthash stratum{};
    stratum.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V1;

    // result[1] is a number, not the expected hex string.
    boost::json::array const result{ boost::json::array{ "mining.notify" }, 1234 };
    stratum.onResponse(makeSubscribeReply(result));

    EXPECT_EQ(1u, stratum.authorizeCount);
}


TEST_F(StratumEthashAuthorizeTest, V1ErrorReplyDoesNotAuthorize)
{
    ProbeStratumEthash stratum{};
    stratum.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V1;

    boost::json::object root;
    root["id"] = stratum::Stratum::ID_MINING_SUBSCRIBE;
    root["error"] = "subscribe rejected";
    stratum.onResponse(root);

    EXPECT_EQ(0u, stratum.authorizeCount);
}


////////////////////////////////////////////////////////////////////////////////
// Other stratum types keep the original ordering: authorize only after a
// parseable extranonce. Guards the fix against leaking outside ETHEREUM_V1.
////////////////////////////////////////////////////////////////////////////////

TEST_F(StratumEthashAuthorizeTest, NonV1ValidResultAuthorizes)
{
    ProbeStratumEthash stratum{};
    stratum.stratumType = stratum::STRATUM_TYPE::ETHPROXY;

    boost::json::array const result{ boost::json::array{ "mining.notify" }, "00ff" };
    stratum.onResponse(makeSubscribeReply(result));

    EXPECT_EQ(1u, stratum.authorizeCount);
    EXPECT_EQ(0x00ffu, stratum.jobInfo.targetBits);
}


TEST_F(StratumEthashAuthorizeTest, NonV1EmptyResultDoesNotAuthorize)
{
    ProbeStratumEthash stratum{};
    stratum.stratumType = stratum::STRATUM_TYPE::ETHPROXY;

    stratum.onResponse(makeSubscribeReply(boost::json::array{}));

    EXPECT_EQ(0u, stratum.authorizeCount);
}
