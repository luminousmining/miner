// Offline protocol coverage for StratumBlake3 (Alephium / woolypooly). No network: the
// handlers parse a JSON message and mutate plain state, so we feed messages directly and
// assert the parsed job / auth state. Locks in the three fixes that once blocked all
// Alephium mining:
//   - authorize replies carry "result" (a bool), not "params";
//   - the per-job share boundary is seeded from targetBlob (no mining.set_difficulty);
//   - the epoch is held constant (the old per-notify bump rebuilt the kernel ~2x/s).

#include <cstdint>
#include <cstring>

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <stratum/blake3.hpp>
#include <stratum/stratum.hpp>


namespace
{
    // targetBlob with non-zero high bytes so toUINT64() (top 8 bytes) is non-zero.
    constexpr char const* TARGET_BLOB{ "00000000ffff0000000000000000000000000000000000000000000000000000" };
    // headerBlob is left-aligned into ubytes[0..]; first byte 0xaa is easy to check.
    constexpr char const* HEADER_BLOB{ "aabbccddeeff00112233445566778899" };


    struct ProbeStratumBlake3 final : public stratum::StratumBlake3
    {
        bool authed() const
        {
            return authenticated;
        }
    };


    boost::json::object makeNotify()
    {
        boost::json::object job;
        job["jobId"] = "deadbeef01";
        job["fromGroup"] = 1u;
        job["toGroup"] = 2u;
        job["targetBlob"] = TARGET_BLOB;
        job["headerBlob"] = HEADER_BLOB;

        boost::json::object root;
        // The handler takes the last element of "params" as the job object.
        root["params"] = boost::json::array{ job };
        return root;
    }
}


struct StratumBlake3ProtocolTest : public testing::Test
{
};


TEST_F(StratumBlake3ProtocolTest, AuthorizeReadsResultBool)
{
    ProbeStratumBlake3 stratum{};

    boost::json::object root;
    root["id"] = stratum::Stratum::ID_MINING_AUTHORIZE;
    root["result"] = true; // note: "result", not "params" (the old bug threw here)
    stratum.onResponse(root);

    EXPECT_TRUE(stratum.authed());
}


TEST_F(StratumBlake3ProtocolTest, AuthorizeResultFalseStaysUnauthenticated)
{
    ProbeStratumBlake3 stratum{};

    boost::json::object root;
    root["id"] = stratum::Stratum::ID_MINING_AUTHORIZE;
    root["result"] = false;
    stratum.onResponse(root);

    EXPECT_FALSE(stratum.authed());
}


TEST_F(StratumBlake3ProtocolTest, NotifyParsesJobAndSeedsBoundaryFromTargetBlob)
{
    ProbeStratumBlake3 stratum{};
    stratum.onMiningNotify(makeNotify());

    EXPECT_EQ("deadbeef01", stratum.jobInfo.jobIDStr);
    EXPECT_EQ(1u, stratum.jobInfo.fromGroup);
    EXPECT_EQ(2u, stratum.jobInfo.toGroup);

    // headerBlob left-aligned: first byte is the first hex pair.
    EXPECT_EQ(0xaau, stratum.jobInfo.headerBlob.ubytes[0]);

    // The share boundary must equal targetBlob (pool sends no set_difficulty).
    algo::hash256 const expected{ algo::toHash256(TARGET_BLOB) };
    EXPECT_EQ(0, std::memcmp(stratum.jobInfo.boundary.ubytes, expected.ubytes, algo::LEN_HASH_256));
    EXPECT_EQ(0, std::memcmp(stratum.jobInfo.boundary.ubytes, stratum.jobInfo.targetBlob.ubytes, algo::LEN_HASH_256));
    EXPECT_NE(0ull, stratum.jobInfo.boundaryU64);
}


TEST_F(StratumBlake3ProtocolTest, NotifyHoldsEpochConstant)
{
    ProbeStratumBlake3 stratum{};

    stratum.onMiningNotify(makeNotify());
    EXPECT_EQ(1, stratum.jobInfo.epoch); // -1 -> 1 on the first job

    stratum.onMiningNotify(makeNotify());
    EXPECT_EQ(1, stratum.jobInfo.epoch); // still 1: no per-notify increment
}
