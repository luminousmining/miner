#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/progpow/meowpow.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/nvidia/meowpow.hpp>
#include <resolver/tests/nvidia.hpp>


struct ResolverMeowpowNvidiaTest : public testing::Test
{
    stratum::StratumJobInfo        jobInfo{};
    common::mocker::MockerStratum  stratum{};
    resolver::tests::Properties    properties{};
    resolver::ResolverNvidiaMeowPOW resolver{};

    ResolverMeowpowNvidiaTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);
        if (false == resolver::tests::initializeCuda(properties))
        {
            logErr() << "Fail init cuda";
        }
        resolver.cuStream = properties.cuStream;
        resolver.cuProperties = &properties.cuProperties;
    }

    ~ResolverMeowpowNvidiaTest()
    {
        resolver::tests::cleanUpCuda();
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.blockNumber = 1118706;
        jobInfo.headerHash = algo::toHash256("44cf248a77ce1623d4fd833ec13ceab83d40591fb9dd7cf7ec28d08f298ba709");
        jobInfo.seedHash = algo::toHash256("cfa3e37c459ebd9b4138bd2141a52d89f6f8f671ecf91456f5a29176eb132fc0");
        jobInfo.boundary = algo::toHash256("0000000500000000000000000000000000000000000000000000000000000000");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.epoch = algo::ethash::findEpoch(jobInfo.seedHash, algo::progpow::EPOCH_LENGTH);
        jobInfo.period = jobInfo.blockNumber / algo::meowpow::MAX_PERIOD;
    }
};


TEST_F(ResolverMeowpowNvidiaTest, findNonce)
{
    initializeJob(0x8071000061a58c77);

    ASSERT_NE(nullptr, resolver.cuStream);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.execute(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0x8071000061a58c77"s, nonceStr);
}


TEST_F(ResolverMeowpowNvidiaTest, aroundFindNonce)
{
    initializeJob(0x8071000061a58c77 - 1024u);

    ASSERT_NE(nullptr, resolver.cuStream);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.execute(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0x8071000061a58c77"s, nonceStr);
}


TEST_F(ResolverMeowpowNvidiaTest, notFindNonce)
{
    initializeJob(0x0071000061a58c77);

    ASSERT_NE(nullptr, resolver.cuStream);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.execute(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmit.empty());
}
