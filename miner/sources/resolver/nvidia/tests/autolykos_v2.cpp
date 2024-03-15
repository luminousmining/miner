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
        resolver.cuStream = properties.cuStream;
    }

    ~ResolverAutolykosv2NvidiaTest()
    {
        resolver::tests::cleanUpCuda();
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.blockNumber = 1034782;
        jobInfo.headerHash = algo::toHash256("d6ff40d44bb470fb3c43b02a67ca3534ff884e2be88484fa89e8c904c0d44392");
        jobInfo.boundary = algo::toHash256("28948096409218832353798863888813816354483909556597628510643976122896");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.period = castU64(algo::autolykos_v2::computePeriod(castU32(jobInfo.blockNumber)));
    }
};


TEST_F(ResolverAutolykosv2NvidiaTest, period)
{
    EXPECT_EQ(104107290u, algo::autolykos_v2::computePeriod(1028992u));
}


TEST_F(ResolverAutolykosv2NvidiaTest, findNonce)
{
    initializeJob(0x5a710000783f4470);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.execute(jobInfo));
    resolver.submit(&stratum);

    EXPECT_FALSE(stratum.paramSubmit.empty());
}
