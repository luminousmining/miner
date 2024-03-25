#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/progpow/kawpow.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/nvidia/kawpow.hpp>
#include <resolver/tests/nvidia.hpp>


struct ResolverKawpowNvidiaTest : public testing::Test
{
    stratum::StratumJobInfo        jobInfo{};
    common::mocker::MockerStratum  stratum{};
    resolver::tests::Properties    properties{};
    resolver::ResolverNvidiaKawPOW resolver{};

    ResolverKawpowNvidiaTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);
        if (false == resolver::tests::initializeCuda(properties))
        {
            logErr() << "Fail init cuda";
        }
        resolver.cuStream = properties.cuStream;
        resolver.cuProperties = &properties.cuProperties;
    }

    ~ResolverKawpowNvidiaTest()
    {
        resolver::tests::cleanUpCuda();
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.blockNumber = 2268417ull;
        jobInfo.headerHash = algo::toHash256("3a35ede9d95b2a36b258013f0175d5975e501ba5ee8d5a86ab5e62ca637d3de9");
        jobInfo.seedHash = algo::toHash256("7a96d7292d380a8ab15178114db4146bf4fede6bf82b126a6ac16ca49e6a5e9f");
        jobInfo.boundary = algo::toHash256("000000027d800000000000000000000000000000000000000000000000000000");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.epoch = algo::ethash::findEpoch(jobInfo.seedHash, algo::progpow::EPOCH_LENGTH);
        jobInfo.period = jobInfo.blockNumber / algo::kawpow::MAX_PERIOD;
    }
};


TEST_F(ResolverKawpowNvidiaTest, findNonce)
{
    initializeJob(0xdec100000704757f);

    ASSERT_NE(nullptr, resolver.cuStream);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.execute(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0xdec100000704757f"s, nonceStr);
}


TEST_F(ResolverKawpowNvidiaTest, notFindNonce)
{
    initializeJob(0x000100000704757f);

    ASSERT_NE(nullptr, resolver.cuStream);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.execute(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmit.empty());
}
