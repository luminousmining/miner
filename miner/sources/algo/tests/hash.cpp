#include <gtest/gtest.h>

#include <algo/hash_utils.hpp>


struct HashTest : public testing::Test
{
    HashTest() = default;
    ~HashTest() = default;
};


TEST_F(HashTest, doubleToHash256)
{
    algo::hash256 const hash256 { algo::toHash256(1.999969) };

    // "80000206012bb4b28a376b73d0f6f86aad62b60d19d23efa40b507f1"
    EXPECT_EQ(0x00, hash256.ubytes[0]);
    EXPECT_EQ(0x00, hash256.ubytes[1]);
    EXPECT_EQ(0x00, hash256.ubytes[2]);
    EXPECT_EQ(0x00, hash256.ubytes[3]);
    EXPECT_EQ(0x80, hash256.ubytes[4]);
    EXPECT_EQ(0x00, hash256.ubytes[5]);
    EXPECT_EQ(0x02, hash256.ubytes[6]);
    EXPECT_EQ(0x06, hash256.ubytes[7]);
    EXPECT_EQ(0x01, hash256.ubytes[8]);
    EXPECT_EQ(0x2b, hash256.ubytes[9]);
    EXPECT_EQ(0xb4, hash256.ubytes[10]);
    EXPECT_EQ(0xb2, hash256.ubytes[11]);
    EXPECT_EQ(0x8a, hash256.ubytes[12]);
    EXPECT_EQ(0x37, hash256.ubytes[13]);
    EXPECT_EQ(0x6b, hash256.ubytes[14]);
    EXPECT_EQ(0x73, hash256.ubytes[15]);
    EXPECT_EQ(0xd0, hash256.ubytes[16]);
    EXPECT_EQ(0xf6, hash256.ubytes[17]);
    EXPECT_EQ(0xf8, hash256.ubytes[18]);
    EXPECT_EQ(0x6a, hash256.ubytes[19]);
    EXPECT_EQ(0xad, hash256.ubytes[20]);
    EXPECT_EQ(0x62, hash256.ubytes[21]);
    EXPECT_EQ(0xb6, hash256.ubytes[22]);
    EXPECT_EQ(0x0d, hash256.ubytes[23]);
    EXPECT_EQ(0x19, hash256.ubytes[24]);
    EXPECT_EQ(0xd2, hash256.ubytes[25]);
    EXPECT_EQ(0x3e, hash256.ubytes[26]);
    EXPECT_EQ(0xfa, hash256.ubytes[27]);
    EXPECT_EQ(0x40, hash256.ubytes[28]);
    EXPECT_EQ(0xb5, hash256.ubytes[29]);
    EXPECT_EQ(0x07, hash256.ubytes[30]);
    EXPECT_EQ(0xf1, hash256.ubytes[31]);
}


TEST_F(HashTest, doubleToHash256MinimunValue)
{
    algo::hash256 const hash256 { algo::toHash256(0.0001) };

    EXPECT_EQ(0x00, hash256.ubytes[0]);
    EXPECT_EQ(0x00, hash256.ubytes[1]);
    EXPECT_EQ(0x27, hash256.ubytes[2]);
    EXPECT_EQ(0x0f, hash256.ubytes[3]);
    EXPECT_EQ(0xd8, hash256.ubytes[4]);
    EXPECT_EQ(0xf0, hash256.ubytes[5]);
    EXPECT_EQ(0x00, hash256.ubytes[6]);
    EXPECT_EQ(0x00, hash256.ubytes[7]);
    EXPECT_EQ(0x00, hash256.ubytes[8]);
    EXPECT_EQ(0x00, hash256.ubytes[9]);
    EXPECT_EQ(0x00, hash256.ubytes[10]);
    EXPECT_EQ(0x00, hash256.ubytes[11]);
    EXPECT_EQ(0x00, hash256.ubytes[12]);
    EXPECT_EQ(0x00, hash256.ubytes[13]);
    EXPECT_EQ(0x00, hash256.ubytes[14]);
    EXPECT_EQ(0x00, hash256.ubytes[15]);
    EXPECT_EQ(0x00, hash256.ubytes[16]);
    EXPECT_EQ(0x00, hash256.ubytes[17]);
    EXPECT_EQ(0x00, hash256.ubytes[18]);
    EXPECT_EQ(0x00, hash256.ubytes[19]);
    EXPECT_EQ(0x00, hash256.ubytes[20]);
    EXPECT_EQ(0x00, hash256.ubytes[21]);
    EXPECT_EQ(0x00, hash256.ubytes[22]);
    EXPECT_EQ(0x00, hash256.ubytes[23]);
    EXPECT_EQ(0x00, hash256.ubytes[24]);
    EXPECT_EQ(0x00, hash256.ubytes[25]);
    EXPECT_EQ(0x00, hash256.ubytes[26]);
    EXPECT_EQ(0x00, hash256.ubytes[27]);
    EXPECT_EQ(0x00, hash256.ubytes[28]);
    EXPECT_EQ(0x00, hash256.ubytes[29]);
    EXPECT_EQ(0x00, hash256.ubytes[30]);
    EXPECT_EQ(0x00, hash256.ubytes[31]);
}


TEST_F(HashTest, doubleToHash256MaxValue)
{
    algo::hash256 const hash256 { algo::toHash256(2.0) };

    EXPECT_EQ(0x00, hash256.ubytes[0]);
    EXPECT_EQ(0x00, hash256.ubytes[1]);
    EXPECT_EQ(0x00, hash256.ubytes[2]);
    EXPECT_EQ(0x00, hash256.ubytes[3]);
    EXPECT_EQ(0x7f, hash256.ubytes[4]);
    EXPECT_EQ(0xff, hash256.ubytes[5]);
    EXPECT_EQ(0x80, hash256.ubytes[6]);
    EXPECT_EQ(0x00, hash256.ubytes[7]);
    EXPECT_EQ(0x00, hash256.ubytes[8]);
    EXPECT_EQ(0x00, hash256.ubytes[9]);
    EXPECT_EQ(0x00, hash256.ubytes[10]);
    EXPECT_EQ(0x00, hash256.ubytes[11]);
    EXPECT_EQ(0x00, hash256.ubytes[12]);
    EXPECT_EQ(0x00, hash256.ubytes[13]);
    EXPECT_EQ(0x00, hash256.ubytes[14]);
    EXPECT_EQ(0x00, hash256.ubytes[15]);
    EXPECT_EQ(0x00, hash256.ubytes[16]);
    EXPECT_EQ(0x00, hash256.ubytes[17]);
    EXPECT_EQ(0x00, hash256.ubytes[18]);
    EXPECT_EQ(0x00, hash256.ubytes[19]);
    EXPECT_EQ(0x00, hash256.ubytes[20]);
    EXPECT_EQ(0x00, hash256.ubytes[21]);
    EXPECT_EQ(0x00, hash256.ubytes[22]);
    EXPECT_EQ(0x00, hash256.ubytes[23]);
    EXPECT_EQ(0x00, hash256.ubytes[24]);
    EXPECT_EQ(0x00, hash256.ubytes[25]);
    EXPECT_EQ(0x00, hash256.ubytes[26]);
    EXPECT_EQ(0x00, hash256.ubytes[27]);
    EXPECT_EQ(0x00, hash256.ubytes[28]);
    EXPECT_EQ(0x00, hash256.ubytes[29]);
    EXPECT_EQ(0x00, hash256.ubytes[30]);
    EXPECT_EQ(0x00, hash256.ubytes[31]);
}
