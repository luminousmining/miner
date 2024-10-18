#include <gtest/gtest.h>

#include <algo/math.hpp>


struct MathTest : public testing::Test
{
    MathTest() = default;
    ~MathTest() = default;
};


TEST_F(MathTest, isOddPrimeTrue)
{
    EXPECT_TRUE(algo::isOddPrime(2ull));
    EXPECT_TRUE(algo::isOddPrime(3ull));
    EXPECT_TRUE(algo::isOddPrime(5ull));
    EXPECT_TRUE(algo::isOddPrime(7ull));
    EXPECT_TRUE(algo::isOddPrime(11ull));
    EXPECT_TRUE(algo::isOddPrime(13ull));
    EXPECT_TRUE(algo::isOddPrime(17ull));
    EXPECT_TRUE(algo::isOddPrime(19ull));
    EXPECT_TRUE(algo::isOddPrime(23ull));
    EXPECT_TRUE(algo::isOddPrime(29ull));
    EXPECT_TRUE(algo::isOddPrime(31ull));
    EXPECT_TRUE(algo::isOddPrime(37ull));
    EXPECT_TRUE(algo::isOddPrime(41ull));
    EXPECT_TRUE(algo::isOddPrime(43ull));
    EXPECT_TRUE(algo::isOddPrime(47ull));
    EXPECT_TRUE(algo::isOddPrime(53ull));
    EXPECT_TRUE(algo::isOddPrime(59ull));
    EXPECT_TRUE(algo::isOddPrime(61ull));
    EXPECT_TRUE(algo::isOddPrime(67ull));
    EXPECT_TRUE(algo::isOddPrime(71ull));
    EXPECT_TRUE(algo::isOddPrime(73ull));
    EXPECT_TRUE(algo::isOddPrime(79ull));
    EXPECT_TRUE(algo::isOddPrime(83ull));
    EXPECT_TRUE(algo::isOddPrime(89ull));
    EXPECT_TRUE(algo::isOddPrime(97ull));
}


TEST_F(MathTest, isOddPrimeFalse)
{
    EXPECT_FALSE(algo::isOddPrime(9ull));
    EXPECT_FALSE(algo::isOddPrime(33ull));
    EXPECT_FALSE(algo::isOddPrime(85ull));
}


TEST_F(MathTest, findPrimeNumber)
{
    EXPECT_EQ(2ull,  algo::largestPrime(2ull));
    EXPECT_EQ(3ull,  algo::largestPrime(3ull));
    EXPECT_EQ(5ull,  algo::largestPrime(5ull));
    EXPECT_EQ(7ull,  algo::largestPrime(7ull));
    EXPECT_EQ(11ull, algo::largestPrime(11ull));
    EXPECT_EQ(13ull, algo::largestPrime(13ull));
    EXPECT_EQ(17ull, algo::largestPrime(17ull));
    EXPECT_EQ(19ull, algo::largestPrime(19ull));
    EXPECT_EQ(23ull, algo::largestPrime(23ull));
    EXPECT_EQ(29ull, algo::largestPrime(29ull));
    EXPECT_EQ(31ull, algo::largestPrime(31ull));
    EXPECT_EQ(37ull, algo::largestPrime(37ull));
    EXPECT_EQ(41ull, algo::largestPrime(41ull));
    EXPECT_EQ(43ull, algo::largestPrime(43ull));
    EXPECT_EQ(47ull, algo::largestPrime(47ull));
    EXPECT_EQ(53ull, algo::largestPrime(53ull));
    EXPECT_EQ(59ull, algo::largestPrime(59ull));
    EXPECT_EQ(61ull, algo::largestPrime(61ull));
    EXPECT_EQ(67ull, algo::largestPrime(67ull));
    EXPECT_EQ(71ull, algo::largestPrime(71ull));
    EXPECT_EQ(73ull, algo::largestPrime(73ull));
    EXPECT_EQ(79ull, algo::largestPrime(79ull));
    EXPECT_EQ(83ull, algo::largestPrime(83ull));
    EXPECT_EQ(89ull, algo::largestPrime(89ull));
    EXPECT_EQ(97ull, algo::largestPrime(97ull));
}


TEST_F(MathTest, findPrimeNumberGap)
{
    EXPECT_EQ(3ull,  algo::largestPrime(5ull - 1ull));
    EXPECT_EQ(5ull,  algo::largestPrime(7ull - 1ull));
    EXPECT_EQ(7ull,  algo::largestPrime(11ull - 1ull));
    EXPECT_EQ(11ull, algo::largestPrime(13ull - 1ull));
    EXPECT_EQ(13ull, algo::largestPrime(17ull - 1ull));
    EXPECT_EQ(17ull, algo::largestPrime(19ull - 1ull));
    EXPECT_EQ(19ull, algo::largestPrime(23ull - 1ull));
    EXPECT_EQ(23ull, algo::largestPrime(29ull - 1ull));
    EXPECT_EQ(29ull, algo::largestPrime(31ull - 1ull));
    EXPECT_EQ(31ull, algo::largestPrime(37ull - 1ull));
    EXPECT_EQ(37ull, algo::largestPrime(41ull - 1ull));
    EXPECT_EQ(41ull, algo::largestPrime(43ull - 1ull));
    EXPECT_EQ(43ull, algo::largestPrime(47ull - 1ull));
    EXPECT_EQ(47ull, algo::largestPrime(53ull - 1ull));
    EXPECT_EQ(53ull, algo::largestPrime(59ull - 1ull));
    EXPECT_EQ(59ull, algo::largestPrime(61ull - 1ull));
    EXPECT_EQ(61ull, algo::largestPrime(67ull - 1ull));
    EXPECT_EQ(67ull, algo::largestPrime(71ull - 1ull));
    EXPECT_EQ(71ull, algo::largestPrime(73ull - 1ull));
    EXPECT_EQ(73ull, algo::largestPrime(79ull - 1ull));
    EXPECT_EQ(79ull, algo::largestPrime(83ull - 1ull));
    EXPECT_EQ(83ull, algo::largestPrime(89ull - 1ull));
    EXPECT_EQ(89ull, algo::largestPrime(97ull - 1ull));
    EXPECT_EQ(97ull, algo::largestPrime(100ull - 1ull));
}
