#include <cstdint>

#include <gtest/gtest.h>

#include <algo/bitwise.hpp>


TEST(Bitwise, nthSetBit)
{
    EXPECT_EQ(0u, algo::nthSetBit(0b1011ull, 0u));
    EXPECT_EQ(1u, algo::nthSetBit(0b1011ull, 1u));
    EXPECT_EQ(3u, algo::nthSetBit(0b1011ull, 2u));
    EXPECT_EQ(64u, algo::nthSetBit(0b1011ull, 3u)); // only 3 bits set
    EXPECT_EQ(64u, algo::nthSetBit(0ull, 0u));
}


TEST(Bitwise, hexToDecimal)
{
    EXPECT_EQ(0xFFull, algo::hexToDecimal<uint64_t>("0xFF"));
    EXPECT_EQ(0xFFull, algo::hexToDecimal<uint64_t>("FF"));
    EXPECT_EQ(0xABCDull, algo::hexToDecimal<uint64_t>("0xabcd"));
    EXPECT_EQ(0ull, algo::hexToDecimal<uint64_t>(""));
    EXPECT_EQ(0ull, algo::hexToDecimal<uint64_t>("xyz"));                                 // invalid -> 0
    EXPECT_EQ(0xFFFFFFFFFFFFFFFFull, algo::hexToDecimal<uint64_t>("0xFFFFFFFFFFFFFFFF")); // 16 hex digits: fits
    EXPECT_EQ(0ull, algo::hexToDecimal<uint64_t>("0x1FFFFFFFFFFFFFFFF"));                 // 17 sig digits: overflow
}


// The template must clamp to each width: a value needing more nibbles than T holds is rejected
// (returns 0), while the widest value that fits is parsed exactly.
TEST(Bitwise, hexToDecimalWidths)
{
    EXPECT_EQ(static_cast<uint8_t>(0xABu), algo::hexToDecimal<uint8_t>("AB"));
    EXPECT_EQ(static_cast<uint8_t>(0u), algo::hexToDecimal<uint8_t>("1AB")); // 3 nibbles: overflow uint8_t

    EXPECT_EQ(static_cast<uint16_t>(0xABCDu), algo::hexToDecimal<uint16_t>("ABCD"));
    EXPECT_EQ(static_cast<uint16_t>(0u), algo::hexToDecimal<uint16_t>("1ABCD")); // overflow uint16_t

    EXPECT_EQ(0xABCD1234u, algo::hexToDecimal<uint32_t>("ABCD1234"));
    EXPECT_EQ(0u, algo::hexToDecimal<uint32_t>("1ABCD1234")); // overflow uint32_t
}
