#include <gtest/gtest.h>

#include <algo/rol.hpp>


struct RotateByteTest : public testing::Test
{
    RotateByteTest() = default;
    ~RotateByteTest() = default;
};


TEST_F(RotateByteTest, rolU64)
{
    EXPECT_EQ(0x440000ull, algo::rol_u64(0x22ull, 0x11ull));
    EXPECT_EQ(0x4400000000ull, algo::rol_u64(0x11ull, 0x22ull));

    EXPECT_EQ(0ull, algo::rol_u64(0x1101ull, 0x2202ull));
    EXPECT_EQ(0ull, algo::rol_u64(0x2202ull, 0x1101ull));
    EXPECT_EQ(0ull, algo::rol_u64(1ull, 0x11111111ull));
}


TEST_F(RotateByteTest, rolU32)
{
    EXPECT_EQ(0x440000u, algo::rol_u32(0x22u, 0x11u));

    EXPECT_EQ(0u, algo::rol_u32(0x11u, 0x22u));
    EXPECT_EQ(0u, algo::rol_u32(0x1101u, 0x2202u));
    EXPECT_EQ(0u, algo::rol_u32(0x2202u, 0x1101u));
    EXPECT_EQ(0u, algo::rol_u32(1u, 0x1111111u));
}               
