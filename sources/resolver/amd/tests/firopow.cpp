#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/progpow/firopow.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/amd/firopow.hpp>
#include <resolver/tests/amd.hpp>


struct ResolverFiropowAmdTest : public testing::Test
{
    stratum::StratumJobInfo       jobInfo{};
    resolver::tests::Properties   properties{};
    resolver::ResolverAmdFiroPOW  resolver{};
    common::mocker::MockerStratum stratum{};

    ResolverFiropowAmdTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);

        if (false == resolver::tests::initializeOpenCL(properties))
        {
            logErr() << "fail init opencl";
        }

        resolver.setDevice(&properties.clDevice);
        resolver.setQueue(&properties.clQueue);
        resolver.setContext(&properties.clContext);
    }

    ~ResolverFiropowAmdTest()
    {
        properties.clDevice = nullptr;
        properties.clContext = nullptr;
        properties.clQueue = nullptr;
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.blockNumber =  544860ull;
        jobInfo.headerHash = algo::toHash256("23bb6340c5ffcf05c3e2b86503a636b805f7ca6c93d50315971b26b72f461790");
        jobInfo.seedHash = algo::toHash256("ac88bf0324754ae04cffe412accbbd72e534acd928bdcbe95239f660667ae26d");
        jobInfo.boundary = algo::toHash256("00000003fffc0000000000000000000000000000000000000000000000000000");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.epoch = algo::ethash::findEpoch(jobInfo.seedHash, algo::firopow::EPOCH_LENGTH);
        jobInfo.period = jobInfo.blockNumber / algo::firopow::MAX_PERIOD;
    }

};


TEST_F(ResolverFiropowAmdTest, findNonce)
{
    initializeJob(0x0000315900f3f0b8);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0x0000315900f3f0b8"s, nonceStr);
}


TEST_F(ResolverFiropowAmdTest, notFindNonce)
{
    initializeJob(0x000100000704757f);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmit.empty());
}
