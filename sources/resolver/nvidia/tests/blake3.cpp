#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/blake3/blake3.hpp>
#include <common/cast.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/nvidia/blake3.hpp>
#include <resolver/tests/nvidia.hpp>


struct ResolverBlake3NvidiaTest : public testing::Test
{
    stratum::StratumJobInfo             jobInfo{};
    resolver::tests::Properties         properties{};
    common::mocker::MockerStratum       stratum{};
    resolver::ResolverNvidiaBlake3      resolver{};

    ResolverBlake3NvidiaTest()
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

    ~ResolverBlake3NvidiaTest()
    {
        resolver::tests::cleanUpCuda(properties);
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.toGroup = 3u;
        jobInfo.fromGroup = 3u;
        jobInfo.headerBlob = algo::toHash<algo::hash3072>(
            "000700000000000022d30e3358af8cd1a732e46d47254bb81ffa43cf402dfe001cf5000000000001bd38686272dfd4b55c3559391adfab12413c197fa92729465aea000000000001ea959d9fbdd9abeda14a65c0040bb7e626d7c13a6e51979a434f000000000002776a5133a5941c8bf35e38043a1f4c5700c0844cb835c414c4300000000000017266826c86467877ca053f0776c56a9d340f634237a91e3865410000000000009dc9c87ce69910b950d0967c9a37930fcfd34f510fc0a014861200000000000157c2d1db2a2af17d3e7560bfc78b270d2b7e65a7fcff312b3b8310249f0949d3463117e80375771047ab3309102f365ddc609b36eaae1363dfceb25811b3902af4dd41490d75b7f79a0478f7f721681c59178a2564b84561559a0000018ea81c4bfa1b029ed6",
            algo::HASH_SHIFT::LEFT);
        logInfo() << "headerblob: " << algo::toHex(jobInfo.headerBlob);
        jobInfo.targetBlob = algo::toHash256("00000001ffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    }
};


TEST_F(ResolverBlake3NvidiaTest, test)
{
    initializeJob(0x914544566c9a0a4d);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));

    resolver.setBlocks(128);
    resolver.setThreads(128);

    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    EXPECT_FALSE(stratum.paramSubmit.empty());
}
