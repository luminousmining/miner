#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/ethash/ethash.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/amd/ethash.hpp>
#include <resolver/tests/amd.hpp>


struct ResolverEthashAmdTest : public testing::Test
{
    stratum::StratumJobInfo       jobInfo{};
    resolver::ResolverAmdEthash   resolver{};
    resolver::tests::Properties   properties{};
    common::mocker::MockerStratum stratum{};

    ResolverEthashAmdTest()
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

    ~ResolverEthashAmdTest()
    {
        properties.clDevice = nullptr;
        properties.clContext = nullptr;
        properties.clQueue = nullptr;
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.headerHash = algo::toHash256("fdfd1516041984564ed59d979b1bb55516c1112dab7ec4b91c861e734abc406a");
        jobInfo.seedHash = algo::toHash256("3c77b17f5e89ebc92dc434fd7d631d80e5b522f07c9a452bc735c05c152381a1");
        jobInfo.boundary = algo::toHash256("0000000225c17d04dad2a2a5b11fb33e22c8af8733a66744290d88c89181a499");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.epoch = algo::ethash::findEpoch(jobInfo.seedHash, algo::ethash::MAX_EPOCH_NUMBER);
    }
};


TEST_F(ResolverEthashAmdTest, emptyJob)
{
    ASSERT_FALSE(resolver.updateMemory(jobInfo));
}


TEST_F(ResolverEthashAmdTest, findNonce)
{
    initializeJob(0x77530000094A7C09);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.execute(jobInfo));
    resolver.submit(&stratum);

    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("77530000094a7c09"s, nonceStr);
}


TEST_F(ResolverEthashAmdTest, notFindNonce)
{
    initializeJob(0x240EB7CA1CF27C09);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.execute(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmit.empty());
}
