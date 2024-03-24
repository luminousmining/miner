#include <gtest/gtest.h>

#include <algo/crypto/kiss99.hpp>


struct Kiss99Test : public testing::Test
{
    Kiss99Test() = default;
    ~Kiss99Test() = default;

    algo::Kiss99Properties build(
        uint32_t const z,
        uint32_t const w,
        uint32_t const jsr,
        uint32_t const jcong)
    {
        algo::Kiss99Properties properties{};

        properties.z = z;
        properties.w = w;
        properties.jsr = jsr;
        properties.jcong = jcong;

        return properties;
    }
};


TEST_F(Kiss99Test, kiss99)
{
    EXPECT_EQ(2437187438u, algo::kiss99(build(1u, 2u, 3u, 4u)));
    EXPECT_EQ(1111177990u, algo::kiss99(build(4u, 3u, 2u, 1u)));
}


// https://github.com/ethereum/EIPs/blob/master/assets/eip-1057/test-vectors.md
TEST_F(Kiss99Test, fromETH)
{
    algo::Kiss99Properties properties { build(362436069u, 521288629u, 123456789u, 380116160u) };

    EXPECT_EQ(769445856u, algo::kiss99(properties));
    EXPECT_EQ(742012328u, algo::kiss99(properties));
    EXPECT_EQ(2121196314u, algo::kiss99(properties));
    EXPECT_EQ(2805620942u, algo::kiss99(properties));

    uint32_t const maxLopp { 100000 - 4 - 1 };
    for (uint32_t i { 0u }; i < maxLopp; ++i)
    {
        algo::kiss99(properties);
    }
    EXPECT_EQ(941074834u, algo::kiss99(properties));
}
