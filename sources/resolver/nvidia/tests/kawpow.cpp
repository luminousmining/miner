#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/progpow/kawpow.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <common/config.hpp>
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
        ////////////////////////////////////////////////////////////////////////////
        common::Config& config{ common::Config::instance() };
        config.deviceAlgorithm.ethashBuildLightCacheCPU = true;

        ////////////////////////////////////////////////////////////////////////////
        common::setLogLevel(common::TYPELOG::__DEBUG);
        if (false == resolver::tests::initializeCuda(properties))
        {
            logErr() << "Fail init cuda";
        }

        ////////////////////////////////////////////////////////////////////////////
        resolver.cuStream[0] = properties.cuStream;
        resolver.cuStream[1] = properties.cuStream;
        resolver.cuProperties = &properties.cuProperties;
        resolver.cuDevice = &properties.cuDevice;
    }

    ~ResolverKawpowNvidiaTest()
    {
        ////////////////////////////////////////////////////////////////////////////
        common::Config& config{ common::Config::instance() };
        config.occupancy.threads = std::nullopt;
        config.occupancy.blocks = std::nullopt;

        ////////////////////////////////////////////////////////////////////////////
        resolver::tests::cleanUpCuda(properties);
    }

    void setOccupancy(
        std::optional<uint32_t> const thread = std::nullopt,
        std::optional<uint32_t> const block = std::nullopt,
        std::optional<uint32_t> const internalLoop = std::nullopt)
    {
        common::Config& config{ common::Config::instance() };
        if (std::nullopt != thread)
        {
            config.occupancy.threads = *thread;
        }
        if (std::nullopt != block)
        {
            config.occupancy.blocks = *block;
        }
        if (std::nullopt != internalLoop)
        {
            config.occupancy.internalLoop = *internalLoop;
        }
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.blockNumber = 965398ull;
        jobInfo.headerHash = algo::toHash256("71c967486cb3b70d5dfcb2ebd8eeef138453637cacbf3ccb580a41a7e96986bb");
        jobInfo.seedHash = algo::toHash256("7c4fb8a5d141973b69b521ce76b0dc50f0d2834d817c7f8310a6ab5becc6bb0c");
        jobInfo.boundary = algo::toHash256("00000000ffff0000000000000000000000000000000000000000000000000000");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.epoch = algo::ethash::ContextGenerator::instance().findEpoch(jobInfo.seedHash, algo::progpow::EPOCH_LENGTH);
        jobInfo.period = jobInfo.blockNumber / algo::kawpow::MAX_PERIOD;
    }
};


TEST_F(ResolverKawpowNvidiaTest, findNonceWithLightCacheGPU)
{
    ////////////////////////////////////////////////////////////////////////////
    common::Config& config{ common::Config::instance() };
    config.deviceAlgorithm.ethashBuildLightCacheCPU = false;

    ////////////////////////////////////////////////////////////////////////////
    initializeJob(0xce00000017f87f70);

    ////////////////////////////////////////////////////////////////////////////
    ASSERT_NE(nullptr, resolver.cuStream[0]);
    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));

    ////////////////////////////////////////////////////////////////////////////
    resolver.submit(&stratum);
    ASSERT_FALSE(stratum.paramSubmit.empty());
    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;
    EXPECT_EQ("0xce00000017f87f7a"s, nonceStr);
}


TEST_F(ResolverKawpowNvidiaTest, findNonceWithLightCacheCPU)
{
    ////////////////////////////////////////////////////////////////////////////
    common::Config& config{ common::Config::instance() };
    config.deviceAlgorithm.ethashBuildLightCacheCPU = true;

    ////////////////////////////////////////////////////////////////////////////
    initializeJob(0xce00000017f87f70);

    ////////////////////////////////////////////////////////////////////////////
    ASSERT_NE(nullptr, resolver.cuStream[0]);
    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));

    ////////////////////////////////////////////////////////////////////////////
    resolver.submit(&stratum);
    ASSERT_FALSE(stratum.paramSubmit.empty());
    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };

    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;
    EXPECT_EQ("0xce00000017f87f7a"s, nonceStr);
}


TEST_F(ResolverKawpowNvidiaTest, findNonceInternalLoop)
{
    ////////////////////////////////////////////////////////////////////////////
    setOccupancy(std::nullopt, std::nullopt, 4u);

    ////////////////////////////////////////////////////////////////////////////
    initializeJob(0xce00000017f87f70);

    ////////////////////////////////////////////////////////////////////////////
    ASSERT_NE(nullptr, resolver.cuStream[0]);
    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));

    ////////////////////////////////////////////////////////////////////////////
    resolver.submit(&stratum);
    ASSERT_FALSE(stratum.paramSubmit.empty());

    ////////////////////////////////////////////////////////////////////////////
    std::string const nonceStr{ stratum.paramSubmit[1].as_string().c_str() };
    using namespace std::string_literals;
    EXPECT_EQ("0xce00000017f87f7a"s, nonceStr);
}


TEST_F(ResolverKawpowNvidiaTest, findNonceLittleOccupancy)
{
    ////////////////////////////////////////////////////////////////////////////
    setOccupancy(32u, 4096u);

    ////////////////////////////////////////////////////////////////////////////
    initializeJob(0xce00000017f87f70);

    ////////////////////////////////////////////////////////////////////////////
    ASSERT_NE(nullptr, resolver.cuStream[0]);
    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));

    ////////////////////////////////////////////////////////////////////////////
    resolver.submit(&stratum);
    ASSERT_FALSE(stratum.paramSubmit.empty());

    ////////////////////////////////////////////////////////////////////////////
    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };
    using namespace std::string_literals;
    EXPECT_EQ("0xce00000017f87f7a"s, nonceStr);
}


TEST_F(ResolverKawpowNvidiaTest, aroundFindNonce)
{
    ////////////////////////////////////////////////////////////////////////////
    initializeJob(0xce00000017f87f70 - 1024u);

    ////////////////////////////////////////////////////////////////////////////
    ASSERT_NE(nullptr, resolver.cuStream[0]);

    ////////////////////////////////////////////////////////////////////////////
    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));

    ////////////////////////////////////////////////////////////////////////////
    resolver.submit(&stratum);
    ASSERT_FALSE(stratum.paramSubmit.empty());

    ////////////////////////////////////////////////////////////////////////////
    std::string const nonceStr { stratum.paramSubmit[1].as_string().c_str() };
    using namespace std::string_literals;
    EXPECT_EQ("0xce00000017f87f7a"s, nonceStr);
}


TEST_F(ResolverKawpowNvidiaTest, notFindNonce)
{
    ////////////////////////////////////////////////////////////////////////////
    initializeJob(0x00000017f87f7a);

    ////////////////////////////////////////////////////////////////////////////
    ASSERT_NE(nullptr, resolver.cuStream[0]);

    ////////////////////////////////////////////////////////////////////////////
    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));

    ////////////////////////////////////////////////////////////////////////////
    resolver.submit(&stratum);

    ////////////////////////////////////////////////////////////////////////////
    EXPECT_TRUE(stratum.paramSubmit.empty());
}
