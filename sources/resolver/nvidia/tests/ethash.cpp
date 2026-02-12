#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/ethash/ethash.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <common/config.hpp>
#include <resolver/nvidia/ethash.hpp>
#include <resolver/tests/nvidia.hpp>


struct ResolverEthashNvidiaTest : public testing::Test
{
    stratum::StratumJobInfo        jobInfo{};
    resolver::tests::Properties    properties{};
    common::mocker::MockerStratum  stratum{};
    resolver::ResolverNvidiaEthash resolver{};

    ResolverEthashNvidiaTest()
    {
        ////////////////////////////////////////////////////////////////////////////
        common::Config& config{ common::Config::instance() };
        config.deviceAlgorithm.ethashBuildLightCacheCPU = true;

        ////////////////////////////////////////////////////////////////////////////
        common::setLogLevel(common::TYPELOG::__DEBUG);

        ////////////////////////////////////////////////////////////////////////////
        if (false == resolver::tests::initializeCuda(properties))
        {
            logErr() << "Fail init cuda";
        }
        resolver.cuStream[0] = properties.cuStream;
        resolver.cuProperties = &properties.cuProperties;
        resolver.cuDevice = &properties.cuDevice;
    }

    ~ResolverEthashNvidiaTest()
    {
        resolver::tests::cleanUpCuda(properties);
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.headerHash = algo::toHash256("fdfd1516041984564ed59d979b1bb55516c1112dab7ec4b91c861e734abc406a");
        jobInfo.seedHash = algo::toHash256("3c77b17f5e89ebc92dc434fd7d631d80e5b522f07c9a452bc735c05c152381a1");
        jobInfo.boundary = algo::toHash256("0000000225c17d04dad2a2a5b11fb33e22c8af8733a66744290d88c89181a499");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.epoch = algo::ethash::ContextGenerator::instance().findEpoch(jobInfo.seedHash, algo::ethash::MAX_EPOCH_NUMBER);
    }
};


TEST_F(ResolverEthashNvidiaTest, emptyJob)
{
    ASSERT_FALSE(resolver.updateMemory(jobInfo));
}


TEST_F(ResolverEthashNvidiaTest, findNonceWithLightCacheGPU)
{
    ////////////////////////////////////////////////////////////////////////////
    common::Config& config{ common::Config::instance() };
    config.deviceAlgorithm.ethashBuildLightCacheCPU = false;

    initializeJob(0x77530000094A7C09);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("77530000094a7c09"s, nonceStr);
}


TEST_F(ResolverEthashNvidiaTest, findNonceWithLightCacheCPU)
{
    initializeJob(0x77530000094A7C09);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("77530000094a7c09"s, nonceStr);
}


TEST_F(ResolverEthashNvidiaTest, notFindNonce)
{
    initializeJob(0x240EB7CA1CF27C09);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmit.empty());
}
