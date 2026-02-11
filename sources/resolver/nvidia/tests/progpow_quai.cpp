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
        resolver.cuStream[0] = properties.cuStream;
        resolver.cuProperties = &properties.cuProperties;
        resolver.cuDevice = &properties.cuDevice;
    }

    ~ResolverProgpowQuaiNvidiaTest()
    {
        resolver::tests::cleanUpCuda(properties);
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.blockNumber = 381378ull;
        jobInfo.headerHash = algo::toHash256("5a203bfaa558803fcb8ddcf0cbe28cde02357a6ab33125f6be19a03802148504");
        jobInfo.seedHash = algo::toHash256("0000000000000000000000000000000000000000000000000000000000000000");
        jobInfo.boundary = algo::toHash256("0000000225c17d04dad2965cc5a02a23e254c0c3f75d9178046aeb27ce1ca574");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.epoch = algo::ethash::findEpoch(jobInfo.seedHash, algo::progpow_quai::EPOCH_LENGTH);
        jobInfo.period = jobInfo.blockNumber / algo::progpow_quai::MAX_PERIOD;
    }
};


TEST_F(ResolverProgpowQuaiNvidiaTest, findNonce)
{
    initializeJob(0xdb4000000566a380);

    ASSERT_NE(nullptr, resolver.cuStream[0]);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0xdb4000000566a380"s, nonceStr);
}


TEST_F(ResolverProgpowQuaiNvidiaTest, aroundFindNonce)
{
    initializeJob(0xdb4000000566a380 - (256ull * 4096ull - 1ull));

    ASSERT_NE(nullptr, resolver.cuStream[0]);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr{ stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0xdb4000000566a380"s, nonceStr);
}


TEST_F(ResolverProgpowQuaiNvidiaTest, notFindNonce)
{
    initializeJob(0xdb4000000566a381);

    ASSERT_NE(nullptr, resolver.cuStream[0]);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmit.empty());
}
