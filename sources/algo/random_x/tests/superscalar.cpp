#include <gtest/gtest.h>

#include <algo/random_x/superscalar.hpp>

#include <cstring>


namespace
{
    static constexpr uint8_t KEY_000[]{ 't', 'e', 's', 't', ' ', 'k', 'e', 'y', ' ', '0', '0', '0' };
    static constexpr uint8_t KEY_001[]{ 't', 'e', 's', 't', ' ', 'k', 'e', 'y', ' ', '0', '0', '1' };
}


struct RandomXSuperscalarTest : public testing::Test
{
    algo::random_x::SuperscalarProgram programs[algo::random_x::SUPERSCALAR_ITERS]{};

    RandomXSuperscalarTest()
    {
        algo::random_x::buildSuperscalarPrograms(KEY_000, programs);
    }

    ~RandomXSuperscalarTest() = default;
};


TEST_F(RandomXSuperscalarTest, allProgramsNonEmpty)
{
    for (uint32_t i{ 0u }; i < algo::random_x::SUPERSCALAR_ITERS; ++i)
    {
        EXPECT_GT(programs[i].size, 0u) << "program " << i;
    }
}


TEST_F(RandomXSuperscalarTest, allProgramsWithinMaxSize)
{
    for (uint32_t i{ 0u }; i < algo::random_x::SUPERSCALAR_ITERS; ++i)
    {
        EXPECT_LE(programs[i].size, algo::random_x::SUPERSCALAR_MAX_INSTRUCTIONS)
            << "program " << i;
    }
}


TEST_F(RandomXSuperscalarTest, allAddressRegsValid)
{
    for (uint32_t i{ 0u }; i < algo::random_x::SUPERSCALAR_ITERS; ++i)
    {
        EXPECT_LT(programs[i].addressReg, 8u) << "program " << i;
    }
}


TEST_F(RandomXSuperscalarTest, allInstructionFieldsValid)
{
    using T = algo::random_x::ScalarInstType;

    for (uint32_t i{ 0u }; i < algo::random_x::SUPERSCALAR_ITERS; ++i)
    {
        for (uint32_t j{ 0u }; j < programs[i].size; ++j)
        {
            algo::random_x::ScalarInst const& instr{ programs[i].instructions[j] };

            ASSERT_LT(static_cast<uint8_t>(instr.type), 10u)
                << "program " << i << " instr " << j << ": invalid type";

            ASSERT_LT(instr.dst, 8u)
                << "program " << i << " instr " << j << ": dst out of range";

            // IROR_C: shift must be 1..63
            if (instr.type == T::IROR_C)
            {
                uint32_t const shift{ instr.imm & 63u };
                EXPECT_NE(0u, shift)
                    << "program " << i << " instr " << j << ": IROR_C shift is 0";
            }

            // IMUL_RCP: divisor must not be 0 or power of 2
            if (instr.type == T::IMUL_RCP)
            {
                EXPECT_NE(0u, instr.imm)
                    << "program " << i << " instr " << j << ": IMUL_RCP imm is 0";
                EXPECT_NE(0u, instr.imm & (instr.imm - 1u))
                    << "program " << i << " instr " << j << ": IMUL_RCP imm is power of 2";
            }
        }
    }
}


TEST_F(RandomXSuperscalarTest, deterministic)
{
    algo::random_x::SuperscalarProgram programs2[algo::random_x::SUPERSCALAR_ITERS]{};
    algo::random_x::buildSuperscalarPrograms(KEY_000, programs2);

    for (uint32_t i{ 0u }; i < algo::random_x::SUPERSCALAR_ITERS; ++i)
    {
        ASSERT_EQ(programs[i].size,       programs2[i].size)       << "program " << i;
        ASSERT_EQ(programs[i].addressReg, programs2[i].addressReg) << "program " << i;
        for (uint32_t j{ 0u }; j < programs[i].size; ++j)
        {
            ASSERT_EQ(static_cast<uint8_t>(programs[i].instructions[j].type),
                      static_cast<uint8_t>(programs2[i].instructions[j].type))
                << "program " << i << " instr " << j;
            ASSERT_EQ(programs[i].instructions[j].dst, programs2[i].instructions[j].dst)
                << "program " << i << " instr " << j;
            ASSERT_EQ(programs[i].instructions[j].src, programs2[i].instructions[j].src)
                << "program " << i << " instr " << j;
            ASSERT_EQ(programs[i].instructions[j].imm, programs2[i].instructions[j].imm)
                << "program " << i << " instr " << j;
        }
    }
}


TEST_F(RandomXSuperscalarTest, keySensitive)
{
    algo::random_x::SuperscalarProgram programs2[algo::random_x::SUPERSCALAR_ITERS]{};
    algo::random_x::buildSuperscalarPrograms(KEY_001, programs2);

    bool allSame{ true };
    for (uint32_t i{ 0u }; i < algo::random_x::SUPERSCALAR_ITERS && allSame; ++i)
    {
        if (programs[i].size != programs2[i].size
            || programs[i].addressReg != programs2[i].addressReg)
        {
            allSame = false;
        }
    }
    EXPECT_FALSE(allSame);
}


TEST_F(RandomXSuperscalarTest, executeDoesNotCrash)
{
    uint64_t r[8]{ 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u };
    for (uint32_t i{ 0u }; i < algo::random_x::SUPERSCALAR_ITERS; ++i)
    {
        algo::random_x::executeSuperscalarProgram(programs[i], r);
    }
    // Result must be non-zero (astronomically unlikely to be all zeros)
    bool anyNonZero{ false };
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        if (0u != r[i]) { anyNonZero = true; break; }
    }
    EXPECT_TRUE(anyNonZero);
}


TEST(RandomXComputeReciprocalTest, zeroForNoOp)
{
    // divisor == 0 or power of 2 → no-op, reciprocal is 0
    EXPECT_EQ(0u, algo::random_x::superscalarComputeReciprocal(0u));
    EXPECT_EQ(0u, algo::random_x::superscalarComputeReciprocal(1u));
    EXPECT_EQ(0u, algo::random_x::superscalarComputeReciprocal(2u));
    EXPECT_EQ(0u, algo::random_x::superscalarComputeReciprocal(4u));
    EXPECT_EQ(0u, algo::random_x::superscalarComputeReciprocal(256u));
    EXPECT_EQ(0u, algo::random_x::superscalarComputeReciprocal(0x80000000u));
}


TEST(RandomXComputeReciprocalTest, nonZeroForValidDivisors)
{
    EXPECT_NE(0u, algo::random_x::superscalarComputeReciprocal(3u));
    EXPECT_NE(0u, algo::random_x::superscalarComputeReciprocal(5u));
    EXPECT_NE(0u, algo::random_x::superscalarComputeReciprocal(7u));
    EXPECT_NE(0u, algo::random_x::superscalarComputeReciprocal(0xFFFFFFFFu));
}


TEST(RandomXComputeReciprocalTest, deterministic)
{
    EXPECT_EQ(
        algo::random_x::superscalarComputeReciprocal(123456789u),
        algo::random_x::superscalarComputeReciprocal(123456789u));
}
