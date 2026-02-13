#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/progpow/progpow_quai.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/amd/progpow_quai.hpp>
#include <resolver/tests/amd.hpp>


struct ResolverProgpowQuaiAmdTest : public testing::Test
{
    stratum::StratumJobInfo          jobInfo{};
    resolver::tests::Properties      properties{};
    resolver::ResolverAmdProgpowQuai resolver{};
    common::mocker::MockerStratum    stratum{};

    ResolverProgpowQuaiAmdTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);
    }

    ~ResolverProgpowQuaiAmdTest()
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
        jobInfo.blockNumber = 381378ull;
        jobInfo.headerHash = algo::toHash256("5a203bfaa558803fcb8ddcf0cbe28cde02357a6ab33125f6be19a03802148504");
        jobInfo.seedHash = algo::toHash256("0000000000000000000000000000000000000000000000000000000000000000");
        jobInfo.boundary = algo::toHash256("0000000225c17d04dad2965cc5a02a23e254c0c3f75d9178046aeb27ce1ca574");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.epoch = algo::ethash::ContextGenerator::instance().findEpoch(jobInfo.seedHash, algo::progpow_quai::EPOCH_LENGTH);
        jobInfo.period = jobInfo.blockNumber / algo::progpow_quai::MAX_PERIOD;
    }
};


TEST_F(ResolverProgpowQuaiAmdTest, findNonce)
{
    initializeDevice(0u);
    initializeJob(0xdb4000000566a380);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr{ stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0xdb4000000566a380"s, nonceStr);
}


TEST_F(ResolverProgpowQuaiAmdTest, aroundFindNonce)
{
    initializeDevice(0u);
    initializeJob(0xdb4000000566a380 - 1024u);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr{ stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0xdb4000000566a380"s, nonceStr);
}


TEST_F(ResolverProgpowQuaiAmdTest, notFindNonce)
{
    initializeDevice(0u);
    initializeJob(0xdb4000000566a381);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmit.empty());
}


TEST_F(ResolverProgpowQuaiAmdTest, allDeviceFindNonce)
{
    uint32_t const countDevice{ resolver::tests::getDeviceCount() };
    for (uint32_t index{ 0u }; index < countDevice; ++index)
    {
        initializeDevice(index);
        initializeJob(0x00000000004cb6fa);

        ASSERT_TRUE(resolver.updateMemory(jobInfo));
        ASSERT_TRUE(resolver.updateConstants(jobInfo));
        ASSERT_TRUE(resolver.executeSync(jobInfo));
        resolver.submit(&stratum);

        ASSERT_FALSE(stratum.paramSubmit.empty());

        std::string const nonceStr{ stratum.paramSubmit[1].as_string().c_str() };

        using namespace std::string_literals;
        EXPECT_EQ("0x00000000004cb6fa"s, nonceStr);
    }
}
