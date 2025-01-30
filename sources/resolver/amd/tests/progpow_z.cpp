#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/progpow/progpow.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/amd/progpow.hpp>
#include <resolver/tests/amd.hpp>


struct ResolverProgpowZAmdTest : public testing::Test
{
    stratum::StratumJobInfo          jobInfo{};
    resolver::tests::Properties      properties{};
    resolver::ResolverAmdProgPOW     resolver{};
    common::mocker::MockerStratum    stratum{};

    ResolverProgpowZAmdTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);
    }

    ~ResolverProgpowZAmdTest()
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
        jobInfo.blockNumber = 3008269ull;
        jobInfo.headerHash = algo::toHash256("0xb1b53f854ef610f73fe4ca24b9d7b04ef9fdb3b11a55a1f9158cbd7c36058fab");
        jobInfo.seedHash = algo::toHash256("0x71a56feffb6f10ea9d76e1a9464eb0abd86e4349ae98fb794923a65b650282a3");
        jobInfo.boundary = algo::toHash256("0x00000002dd01fc067918c87bb22ae2da831babaff47cc1f55b39398a682631f0");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.epoch = algo::ethash::findEpoch(jobInfo.seedHash, algo::ethash::EPOCH_LENGTH);
        jobInfo.period = jobInfo.blockNumber / algo::progpow::v_0_9_2::MAX_PERIOD;
    }
};


TEST_F(ResolverProgpowZAmdTest, findNonce)
{
    initializeDevice(0u);
    initializeJob(0x9f990000004cb6fa);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr{ stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0x9f990000004cb6fa"s, nonceStr);
}


TEST_F(ResolverProgpowZAmdTest, aroundFindNonce)
{
    initializeDevice(0u);
    initializeJob(0x62114e8a70455eef - 1024u);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr{ stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0x62114e8a70455eef"s, nonceStr);
}


TEST_F(ResolverProgpowZAmdTest, notFindNonce)
{
    initializeDevice(0u);
    initializeJob(0ull);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmit.empty());
}


TEST_F(ResolverProgpowZAmdTest, allDeviceFindNonce)
{
    uint32_t const countDevice{ resolver::tests::getDeviceCount() };
    for (uint32_t index{ 0u }; index < countDevice; ++index)
    {
        initializeDevice(index);
        initializeJob(0x62114e8a70455eef);

        ASSERT_TRUE(resolver.updateMemory(jobInfo));
        ASSERT_TRUE(resolver.updateConstants(jobInfo));
        ASSERT_TRUE(resolver.executeSync(jobInfo));
        resolver.submit(&stratum);

        ASSERT_FALSE(stratum.paramSubmit.empty());

        std::string const nonceStr{ stratum.paramSubmit[1].as_string().c_str() };

        using namespace std::string_literals;
        EXPECT_EQ("0x62114e8a70455eef"s, nonceStr);
    }
}
