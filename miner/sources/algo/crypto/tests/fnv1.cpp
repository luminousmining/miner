#include <gtest/gtest.h>

#include <algo/crypto/fnv1.hpp>


struct Fnv1est : public testing::Test
{
    Fnv1est() = default;
    ~Fnv1est() = default;
};


TEST_F(Fnv1est, fnv1)
{
    EXPECT_EQ(1677762180u, algo::fnv1(100u, 1000u));
    EXPECT_EQ(3892717148u, algo::fnv1(1000u, 100u));

    EXPECT_EQ(3355524360u, algo::fnv1(200u, 2000u));
    EXPECT_EQ(3490467000u, algo::fnv1(2000u, 200u));
}


// https://github.com/ethereum/EIPs/blob/master/assets/eip-1057/test-vectors.md
TEST_F(Fnv1est, fromETH)
{
    EXPECT_EQ(0xD8DCF964, algo::fnv1(0x811C9DC5, 0xDDD0A47B));
    EXPECT_EQ(0xE4F472A8, algo::fnv1(0XD37EE61A, 0XEE304846));
    EXPECT_EQ(0xA9155BBC, algo::fnv1(0XDEDC7AD4, 0X00000000));
}


TEST_F(Fnv1est, fnv1a)
{
    EXPECT_EQ(2349176164u, algo::fnv1a(100u, 1000u));
    EXPECT_EQ(2349176164u, algo::fnv1a(1000u, 100u));
    EXPECT_EQ(algo::fnv1a(100u, 1000u), algo::fnv1a(1000u, 100u));

    EXPECT_EQ(403385032u, algo::fnv1a(200u, 2000u));
    EXPECT_EQ(403385032u, algo::fnv1a(2000u, 200u));
    EXPECT_EQ(algo::fnv1a(200u, 2000u), algo::fnv1a(2000u, 200u));
}
