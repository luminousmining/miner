#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/progpow/progpow.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/nvidia/progpow.hpp>
#include <resolver/tests/nvidia.hpp>


struct ResolverProgpowZNvidiaTest : public testing::Test
{
    stratum::StratumJobInfo             jobInfo{};
    common::mocker::MockerStratum       stratum{};
    resolver::tests::Properties         properties{};
    resolver::ResolverNvidiaProgPOW     resolver{};

    ResolverProgpowZNvidiaTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);
        if (false == resolver::tests::initializeCuda(properties))
        {
            logErr() << "Fail init cuda";
        }
        resolver.cuStream[0] = properties.cuStream;
        resolver.cuProperties = &properties.cuProperties;
        resolver.cuDevice = &properties.cuDevice;
    }

    ~ResolverProgpowZNvidiaTest()
    {
        resolver::tests::cleanUpCuda();
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.blockNumber = std::strtoull("0x00000000002de70d", nullptr, 16);
        jobInfo.headerHash = algo::toHash256("0xb1b53f854ef610f73fe4ca24b9d7b04ef9fdb3b11a55a1f9158cbd7c36058fab");
        jobInfo.seedHash = algo::toHash256("0x71a56feffb6f10ea9d76e1a9464eb0abd86e4349ae98fb794923a65b650282a3");
        jobInfo.boundary = algo::toHash256("0x00000002dd01fc067918c87bb22ae2da831babaff47cc1f55b39398a682631f0");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.epoch = cast32(jobInfo.blockNumber / castU64(algo::ethash::EPOCH_LENGTH));
        jobInfo.period = jobInfo.blockNumber / algo::progpow::v_0_9_2::MAX_PERIOD;
    }
};


TEST_F(ResolverProgpowZNvidiaTest, jobValid)
{
    initializeJob(0x62114e8a70455eef);

    ASSERT_EQ(3008269ull, jobInfo.blockNumber);
    ASSERT_EQ(100u, jobInfo.epoch);
    ASSERT_EQ(60165, jobInfo.period);
}


TEST_F(ResolverProgpowZNvidiaTest, findNonce)
{
    initializeJob(0x62114e8a70455eef);

    ASSERT_NE(nullptr, resolver.cuStream[0]);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0x62114e8a70455eef"s, nonceStr);
}


TEST_F(ResolverProgpowZNvidiaTest, aroundFindNonce)
{
    initializeJob(0x62114e8a70455eef - 1024u);

    ASSERT_NE(nullptr, resolver.cuStream[0]);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr{ stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0x62114e8a70455eef"s, nonceStr);
}


TEST_F(ResolverProgpowZNvidiaTest, notFindNonce)
{
    initializeJob(0ull);

    ASSERT_NE(nullptr, resolver.cuStream[0]);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmit.empty());
}
