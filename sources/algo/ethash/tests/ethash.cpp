#include <gtest/gtest.h>

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
