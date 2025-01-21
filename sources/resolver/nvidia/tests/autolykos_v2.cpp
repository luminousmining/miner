#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/autolykos/autolykos.hpp>
#include <common/cast.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/nvidia/autolykos_v2.hpp>
#include <resolver/tests/nvidia.hpp>


struct ResolverAutolykosv2NvidiaTest : public testing::Test
{
    stratum::StratumJobInfo             jobInfo{};
    resolver::tests::Properties         properties{};
    common::mocker::MockerStratum       stratum{};
    resolver::ResolverNvidiaAutolykosV2 resolver{};

    ResolverAutolykosv2NvidiaTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);
        if (false == resolver::tests::initializeCuda(properties))
        {
            logErr() << "Fail init cuda";
        }
        resolver.cuStream[0] = properties.cuStream;
        resolver.cuProperties = &properties.cuProperties;
        resolver.cuDevice = &properties.cuDevice;
    }

    ~ResolverAutolykosv2NvidiaTest()
    {
        resolver::tests::cleanUpCuda();
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.blockNumber = 1415098u;
        jobInfo.headerHash = algo::toHash256("6f109ba5226d1e0814cdeec79f1231d1d48196b5979a6d816e3621a1ef47ad80");
        jobInfo.boundary =
        algo::toHash2<algo::hash256, algo::hash512>(
            algo::toLittleEndian<algo::hash512>(
                algo::decimalToHash<algo::hash512>(
                    "28948022309329048855892746252171976963209391069768726095651290785380")));
        jobInfo.period = castU64(algo::autolykos_v2::computePeriod(castU32(jobInfo.blockNumber)));
    }
};


TEST_F(ResolverAutolykosv2NvidiaTest, period)
{
    EXPECT_EQ(104107290u, algo::autolykos_v2::computePeriod(1028992u));
}


TEST_F(ResolverAutolykosv2NvidiaTest, notFindNonce)
{
    initializeJob(11055774136181864435ull);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    EXPECT_FALSE(stratum.paramSubmit.empty());
}


TEST_F(ResolverAutolykosv2NvidiaTest, findNonce)
{
    initializeJob(11055774138563218679ull);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(
        algo::autolykos_v2::mhssamadani::isValidShare(
            jobInfo.headerHash,
            jobInfo.boundary,
            11055774138563218679ull,
            3130463488u));

    EXPECT_FALSE(stratum.paramSubmit.empty());
}
