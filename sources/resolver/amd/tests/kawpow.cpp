#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/progpow/kawpow.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/amd/kawpow.hpp>
#include <resolver/tests/amd.hpp>


struct ResolverKawpowAmdTest : public testing::Test
{
    stratum::StratumJobInfo       jobInfo{};
    resolver::tests::Properties   properties{};
    resolver::ResolverAmdKawPOW   resolver{};
    common::mocker::MockerStratum stratum{};

    ResolverKawpowAmdTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);
    }

    ~ResolverKawpowAmdTest()
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
        jobInfo.blockNumber = 965398ull;
        jobInfo.headerHash = algo::toHash256("71c967486cb3b70d5dfcb2ebd8eeef138453637cacbf3ccb580a41a7e96986bb");
        jobInfo.seedHash = algo::toHash256("7c4fb8a5d141973b69b521ce76b0dc50f0d2834d817c7f8310a6ab5becc6bb0c");
        jobInfo.boundary = algo::toHash256("00000000ffff0000000000000000000000000000000000000000000000000000");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.epoch = algo::ethash::findEpoch(jobInfo.seedHash, algo::progpow::EPOCH_LENGTH);
        jobInfo.period = jobInfo.blockNumber / algo::kawpow::MAX_PERIOD;
    }

};


TEST_F(ResolverKawpowAmdTest, findNonce)
{
    initializeDevice(0u);
    initializeJob(0xce00000017f87f70);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0xce00000017f87f7a"s, nonceStr);
}


TEST_F(ResolverKawpowAmdTest, aroundFindNonce)
{
    initializeDevice(0u);
    initializeJob(0xce00000017f87f70 - 1024u);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0xce00000017f87f7a"s, nonceStr);
}


TEST_F(ResolverKawpowAmdTest, notFindNonce)
{
    initializeDevice(0u);
    initializeJob(0x00000017f87f7a);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmit.empty());
}


TEST_F(ResolverKawpowAmdTest, allDeviceFindNonce)
{
    uint32_t const countDevice { resolver::tests::getDeviceCount() };
    for (uint32_t index { 0u }; index < countDevice; ++index)
    {
        initializeDevice(index);
        initializeJob(0xce00000017f87f70);

        ASSERT_TRUE(resolver.updateMemory(jobInfo));
        ASSERT_TRUE(resolver.updateConstants(jobInfo));
        ASSERT_TRUE(resolver.executeSync(jobInfo));
        resolver.submit(&stratum);

        ASSERT_FALSE(stratum.paramSubmit.empty());

        std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

        using namespace std::string_literals;
        EXPECT_EQ("0xce00000017f87f7a"s, nonceStr);
    }
}
