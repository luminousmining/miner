#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/progpow/progpow_quai.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/nvidia/progpow_quai.hpp>
#include <resolver/tests/nvidia.hpp>


struct ResolverProgpowQuaiNvidiaTest : public testing::Test
{
    stratum::StratumJobInfo             jobInfo{};
    common::mocker::MockerStratum       stratum{};
    resolver::tests::Properties         properties{};
    resolver::ResolverNvidiaProgpowQuai resolver{};

    ResolverProgpowQuaiNvidiaTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);
        if (false == resolver::tests::initializeCuda(properties))
        {
            logErr() << "Fail init cuda";
        }
        resolver.cuStream = properties.cuStream;
        resolver.cuProperties = &properties.cuProperties;
    }

    ~ResolverProgpowQuaiNvidiaTest()
    {
        resolver::tests::cleanUpCuda();
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.blockNumber = 48263ull;
        jobInfo.headerHash = algo::toHash256("ff4fae1e1d8ae09e787b6841cf31db63be783346db3ce6dab388090d7b37f4a2");
        jobInfo.seedHash = algo::toHash256("0000000000000000000000000000000000000000000000000000000000000000");
        jobInfo.boundary = algo::toHash256("0000000394427b08175efa9a9eb59b9123e2969bf19bf272b20787ed022fbe6c");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.epoch = algo::ethash::findEpoch(jobInfo.seedHash, algo::progpow_quai::EPOCH_LENGTH);
        jobInfo.period = jobInfo.blockNumber / algo::progpow_quai::MAX_PERIOD;
    }
};


TEST_F(ResolverProgpowQuaiNvidiaTest, findNonce)
{
    initializeJob(0x9f990000004cb6fa);

    ASSERT_NE(nullptr, resolver.cuStream);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.execute(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0x9f990000004cb6fa"s, nonceStr);
}


TEST_F(ResolverProgpowQuaiNvidiaTest, aroundFindNonce)
{
    initializeJob(0x9f990000004cb6fa - 1024u);

    ASSERT_NE(nullptr, resolver.cuStream);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.execute(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr{ stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0x9f990000004cb6fa"s, nonceStr);
}


TEST_F(ResolverProgpowQuaiNvidiaTest, notFindNonce)
{
    initializeJob(0x00000000004cb6fa);

    ASSERT_NE(nullptr, resolver.cuStream);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.execute(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmit.empty());
}
