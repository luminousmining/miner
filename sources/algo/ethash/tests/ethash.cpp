#include <gtest/gtest.h>

#include <algo/dag_context.hpp>
#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/ethash/ethash.hpp>


struct EthashTest : public testing::Test
{
    std::array<algo::hash256, 10> hashes
    {
        /*0*/ algo::toHash256("0000000000000000000000000000000000000000000000000000000000000000"),
        /*1*/ algo::toHash256("290decd9548b62a8d60345a988386fc84ba6bc95484008f6362f93160ef3e563"),
        /*2*/ algo::toHash256("a9b0e0c9aca72c07ba06b5bbdae8b8f69e61878301508473379bb4f71807d707"),
        /*3*/ algo::toHash256("9a845f3f6ff2c8b20f8528e2af91156778be0b0a49030cbf2c58a86cc4462f17"),
        /*4*/ algo::toHash256("f652f3eec5cc0db7a1d1cd053f4eabf7db6403c0d203b50ce9725035596b1d06"),
        /*5*/ algo::toHash256("20a7678ca7b50829183baac2e1e3c43fa3c4bcbc171b11cf5a9f30bebd172920"),
        /*6*/ algo::toHash256("1222b1faed7f93098f8ae498621fb3479805a664b70186063861c46596c66164"),
        /*7*/ algo::toHash256("ee1d0f61b054dff0f3025ebba821d405c8dc19a983e582e9fa5436fc3e7a07d8"),
        /*8*/ algo::toHash256("9472a82f992649315e3977120843a5a246e375715bd70ee98b3dd77c63154e99"),
        /*9*/ algo::toHash256("09b435f2d92d0ddee038c379be8db1f895c904282e9ceb790f519a6aa3f83810")
    };

    EthashTest() = default;
    ~EthashTest() = default;
};


TEST_F(EthashTest, epoch)
{
    int32_t const maxEpochNumber { cast32(algo::ethash::MAX_EPOCH_NUMBER) };

    EXPECT_EQ(0,                  algo::ethash::findEpoch(hashes[0], maxEpochNumber));
    EXPECT_EQ(1,                  algo::ethash::findEpoch(hashes[1], maxEpochNumber));
    EXPECT_EQ(171,                algo::ethash::findEpoch(hashes[2], maxEpochNumber));
    EXPECT_EQ(604,                algo::ethash::findEpoch(hashes[3], maxEpochNumber));
    EXPECT_EQ(588,                algo::ethash::findEpoch(hashes[4], maxEpochNumber));
    EXPECT_EQ(2048,               algo::ethash::findEpoch(hashes[5], maxEpochNumber));
    EXPECT_EQ(29998,              algo::ethash::findEpoch(hashes[6], maxEpochNumber));
    EXPECT_EQ(29999,              algo::ethash::findEpoch(hashes[7], maxEpochNumber));
    EXPECT_EQ(maxEpochNumber - 1, algo::ethash::findEpoch(hashes[8], maxEpochNumber));
    EXPECT_EQ(maxEpochNumber,     algo::ethash::findEpoch(hashes[9], maxEpochNumber));
}


TEST_F(EthashTest, lightCacheBuild)
{
    algo::DagContext context{};
    uint32_t maxEpoch{ algo::ethash::MAX_EPOCH_NUMBER };
    uint32_t lightCacheCountItemsGrowth{ algo::ethash::LIGHT_CACHE_COUNT_ITEMS_GROWTH };
    uint32_t lightCacheCountItemsInit{ algo::ethash::LIGHT_CACHE_COUNT_ITEMS_INIT };
    uint32_t dagCountItemsGrowth{ algo::ethash::DAG_COUNT_ITEMS_GROWTH };
    uint32_t dagCountItemsInit{ algo::ethash::DAG_COUNT_ITEMS_INIT };

    algo::ethash::initializeDagContext
    (
        context,
        561ull,
        maxEpoch,
        dagCountItemsGrowth,
        dagCountItemsInit,
        lightCacheCountItemsGrowth,
        lightCacheCountItemsInit,
        true
    );

    // light cache size
    ASSERT_EQ(1411061ull,    context.lightCache.numberItem);
    ASSERT_EQ(90307904ull,   context.lightCache.size);

    // dag cache size
    ASSERT_EQ(45154283ull,   context.dagCache.numberItem);
    ASSERT_EQ(5779748224ull, context.dagCache.size);

    // [0-9]
    ASSERT_EQ(2305997711u,  context.lightCache.hash[0].word32[0]);
    ASSERT_EQ(1487327327u,  context.lightCache.hash[1].word32[0]);
    ASSERT_EQ(2092739119u,  context.lightCache.hash[2].word32[0]);
    ASSERT_EQ(3128643576u,  context.lightCache.hash[3].word32[0]);
    ASSERT_EQ(760532322u,   context.lightCache.hash[4].word32[0]);
    ASSERT_EQ(3333110563u,  context.lightCache.hash[5].word32[0]);
    ASSERT_EQ(995681143u,   context.lightCache.hash[6].word32[0]);
    ASSERT_EQ(3902161905u,  context.lightCache.hash[7].word32[0]);
    ASSERT_EQ(1006606786u,  context.lightCache.hash[8].word32[0]);
    ASSERT_EQ(2727137495u,  context.lightCache.hash[9].word32[0]);

    // [100000-100009]
    ASSERT_EQ(2745248448u,  context.lightCache.hash[100000].word32[0]);
    ASSERT_EQ(3529564367u,  context.lightCache.hash[100001].word32[0]);
    ASSERT_EQ(3190066050u,  context.lightCache.hash[100002].word32[0]);
    ASSERT_EQ(1735868306u,  context.lightCache.hash[100003].word32[0]);
    ASSERT_EQ(1257913133u,  context.lightCache.hash[100004].word32[0]);
    ASSERT_EQ(816297627u,   context.lightCache.hash[100005].word32[0]);
    ASSERT_EQ(4195235099u,  context.lightCache.hash[100006].word32[0]);
    ASSERT_EQ(1584650964u,  context.lightCache.hash[100007].word32[0]);
    ASSERT_EQ(725852591u,   context.lightCache.hash[100008].word32[0]);
    ASSERT_EQ(2492012866u,  context.lightCache.hash[100009].word32[0]);

    // [1000000-1000009]
    ASSERT_EQ(3340744594u,  context.lightCache.hash[1000000].word32[0]);
    ASSERT_EQ(3027713121u,  context.lightCache.hash[1000001].word32[0]);
    ASSERT_EQ(1612654679u,  context.lightCache.hash[1000002].word32[0]);
    ASSERT_EQ(1424449202u,  context.lightCache.hash[1000003].word32[0]);
    ASSERT_EQ(3163348883u,  context.lightCache.hash[1000004].word32[0]);
    ASSERT_EQ(1928801235u,  context.lightCache.hash[1000005].word32[0]);
    ASSERT_EQ(2487024762u,  context.lightCache.hash[1000006].word32[0]);
    ASSERT_EQ(541218828u,   context.lightCache.hash[1000007].word32[0]);
    ASSERT_EQ(762008954u,   context.lightCache.hash[1000008].word32[0]);
    ASSERT_EQ(311220328u,   context.lightCache.hash[1000009].word32[0]);

    algo::ethash::freeDagContext(context);
}
