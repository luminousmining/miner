#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/progpow/firopow.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/nvidia/firopow.hpp>
#include <resolver/tests/nvidia.hpp>


struct ResolverFiropowNvidiaTest : public testing::Test
{
    stratum::StratumJobInfo         jobInfo{};
    common::mocker::MockerStratum   stratum{};
    resolver::tests::Properties     properties{};
    resolver::ResolverNvidiaFiroPOW resolver{};

    ResolverFiropowNvidiaTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);
        if (false == resolver::tests::initializeCuda(properties))
        {
            logErr() << "Fail init cuda";
        }
        resolver.cuStream[0] = properties.cuStream;
        resolver.cuProperties = &properties.cuProperties;
    }

    ~ResolverFiropowNvidiaTest()
    {
        resolver::tests::cleanUpCuda();
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


TEST_F(ResolverFiropowNvidiaTest, findNonce)
{
    initializeJob(0x0000315900f3f0b8);

    ASSERT_NE(nullptr, resolver.cuStream[0]);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0x0000315900f3f0b8"s, nonceStr);
}


TEST_F(ResolverFiropowNvidiaTest, notFindNonce)
{
    initializeJob(0x000100000704757f);

    ASSERT_NE(nullptr, resolver.cuStream[0]);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmit.empty());
}
