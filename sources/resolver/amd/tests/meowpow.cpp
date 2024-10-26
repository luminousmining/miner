#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/progpow/meowpow.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/amd/meowpow.hpp>
#include <resolver/tests/amd.hpp>


struct ResolverMeowpowAmdTest : public testing::Test
{
    stratum::StratumJobInfo       jobInfo{};
    resolver::tests::Properties   properties{};
    resolver::ResolverAmdMeowPOW  resolver{};
    common::mocker::MockerStratum stratum{};

    ResolverMeowpowAmdTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);
    }

    ~ResolverMeowpowAmdTest()
    {
        properties.clDevice = nullptr;
        properties.clContext = nullptr;
        properties.clQueue = nullptr;
    }

    void initializeDevice(uint32_t const index)
    {
        if (false == resolver::tests::initializeOpenCL(properties, index))
        {
            logErr() << "fail init opencl";
        }

        resolver.setDevice(&properties.clDevice);
        resolver.setQueue(&properties.clQueue);
        resolver.setContext(&properties.clContext);
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.blockNumber = 1118706ull;
        jobInfo.headerHash = algo::toHash256("44cf248a77ce1623d4fd833ec13ceab83d40591fb9dd7cf7ec28d08f298ba709");
        jobInfo.seedHash = algo::toHash256("cfa3e37c459ebd9b4138bd2141a52d89f6f8f671ecf91456f5a29176eb132fc0");
        jobInfo.boundary = algo::toHash256("0000000500000000000000000000000000000000000000000000000000000000");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.epoch = algo::ethash::findEpoch(jobInfo.seedHash, algo::progpow::EPOCH_LENGTH);
        jobInfo.period = jobInfo.blockNumber / algo::meowpow::MAX_PERIOD;
    }
};


TEST_F(ResolverMeowpowAmdTest, findNonce)
{
    initializeDevice(0u);
    initializeJob(0x8071000061a58c77);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.execute(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0x8071000061a58c77"s, nonceStr);
}


TEST_F(ResolverMeowpowAmdTest, aroundFindNonce)
{
    initializeDevice(0u);
    initializeJob(0x8071000061a58c77 - 1024u);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.execute(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0x8071000061a58c77"s, nonceStr);
}


TEST_F(ResolverMeowpowAmdTest, notFindNonce)
{
    initializeDevice(0u);
    initializeJob(0x0071000061a58c77);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.execute(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmit.empty());
}


TEST_F(ResolverMeowpowAmdTest, allDeviceFindNonce)
{
    uint32_t const countDevice { resolver::tests::getDeviceCount() };
    for (uint32_t index { 0u }; index < countDevice; ++index)
    {
        initializeDevice(index);
        initializeJob(0x8071000061a58c77);

        ASSERT_TRUE(resolver.updateMemory(jobInfo));
        ASSERT_TRUE(resolver.updateConstants(jobInfo));
        ASSERT_TRUE(resolver.execute(jobInfo));
        resolver.submit(&stratum);

        ASSERT_FALSE(stratum.paramSubmit.empty());

        std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

        using namespace std::string_literals;
        EXPECT_EQ("0x8071000061a58c77"s, nonceStr);
    }
}
