#if defined(__linux__)
#include <experimental/filesystem>
namespace __fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace __fs = std::filesystem;
#endif
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <algo/progpow/evrprogpow.hpp>
#include <algo/progpow/firopow.hpp>
#include <algo/progpow/kawpow.hpp>
#include <algo/progpow/meowpow.hpp>
#include <algo/progpow/progpow.hpp>
#include <algo/progpow/progpow_quai.hpp>


// Hermetic (no-GPU) guard for the progpow version <-> counts wiring.
//
// `writeMathRandomKernelOpenCL` emits the per-period `sequence_dynamic()` OpenCL
// source: one cache op for each of `countCache` iterations and one math op for
// each of `countMath` iterations. PR #144 shipped a resolver that declared
// VERSION::V_0_9_2 but initialized countCache/countMath to the v_0_9_3 values
// (11/18 instead of 12/20), so the generated kernel had the wrong number of
// operations and computed a different hash than the reference. Nothing
// CI-runnable guarded that -- the only coverage was the AMD device test. These
// tests pin the generator's op counts so a version/counts mismatch is caught on
// any host, no GPU required.


namespace
{
    std::string generatedKernelOpCounts(
        algo::progpow::VERSION const version,
        uint32_t const               deviceId,
        uint64_t const               period,
        uint32_t const               countCache,
        uint32_t const               countMath)
    {
        algo::progpow::writeMathRandomKernelOpenCL(
            version,
            deviceId,
            period,
            countCache,
            countMath,
            algo::progpow::REGS,
            algo::progpow::MODULE_SOURCE);

        __fs::path path{ "kernel" };
        path /= "progpow";
        path /= "sequence_math_random_" + std::to_string(deviceId) + "_" + std::to_string(period) + ".cl";

        std::ifstream     ifs{ path };
        std::stringstream ss;
        ss << ifs.rdbuf();
        ifs.close();

        __fs::remove(path);

        return ss.str();
    }


    size_t countOccurrences(std::string const& haystack, std::string const& needle)
    {
        size_t count{ 0u };
        size_t pos{ 0u };
        while ((pos = haystack.find(needle, pos)) != std::string::npos)
        {
            ++count;
            pos += needle.size();
        }
        return count;
    }


    // One cache op emits exactly one `dag_offset = hash[...]` statement; one math
    // op emits exactly one `sel_math` comment; each DAG_LOAD emits one
    // `merge_entries`. The `uint dag_offset;` declaration does not match the
    // assignment needle.
    size_t cacheOpCount(std::string const& src)
    {
        return countOccurrences(src, "dag_offset = hash[");
    }


    size_t mathOpCount(std::string const& src)
    {
        return countOccurrences(src, "sel_math");
    }


    size_t mergeEntriesCount(std::string const& src)
    {
        return countOccurrences(src, "merge_entries");
    }
}


struct ProgpowKernelOpenCLTest : public testing::Test
{
    ProgpowKernelOpenCLTest()  = default;
    ~ProgpowKernelOpenCLTest() = default;
};


// The constants the base progpow/progpowz resolvers pull from. This is the exact
// value that PR #144 got wrong (it used the v_0_9_3 namespace's 11/18).
TEST_F(ProgpowKernelOpenCLTest, v092CanonicalConstants)
{
    EXPECT_EQ(12u, algo::progpow::v_0_9_2::COUNT_CACHE);
    EXPECT_EQ(20u, algo::progpow::v_0_9_2::COUNT_MATH);
}


// V_0_9_2 with its canonical counts must emit a kernel with exactly 12 cache ops,
// 20 math ops, and DAG_LOADS merge_entries.
TEST_F(ProgpowKernelOpenCLTest, v092EmitsCanonicalOpCounts)
{
    std::string const src{ generatedKernelOpCounts(
        algo::progpow::VERSION::V_0_9_2,
        0u,
        12345ull,
        algo::progpow::v_0_9_2::COUNT_CACHE,
        algo::progpow::v_0_9_2::COUNT_MATH) };

    ASSERT_FALSE(src.empty());
    EXPECT_EQ(algo::progpow::v_0_9_2::COUNT_CACHE, cacheOpCount(src));
    EXPECT_EQ(algo::progpow::v_0_9_2::COUNT_MATH, mathOpCount(src));
    EXPECT_EQ(static_cast<size_t>(algo::progpow::DAG_LOADS), mergeEntriesCount(src));
}


// The #144 signature: feeding the v_0_9_3 counts (the bug) to a V_0_9_2 kernel
// yields a structurally different kernel -- proof the mismatch produced a wrong
// kernel, and that the generator is sensitive to the counts (not ignoring them).
TEST_F(ProgpowKernelOpenCLTest, mismatchedCountsChangeKernel)
{
    std::string const canonical{ generatedKernelOpCounts(
        algo::progpow::VERSION::V_0_9_2,
        1u,
        12345ull,
        algo::progpow::v_0_9_2::COUNT_CACHE,
        algo::progpow::v_0_9_2::COUNT_MATH) };

    std::string const buggy{ generatedKernelOpCounts(
        algo::progpow::VERSION::V_0_9_2,
        2u,
        12345ull,
        algo::progpow::v_0_9_3::COUNT_CACHE,
        algo::progpow::v_0_9_3::COUNT_MATH) };

    EXPECT_EQ(algo::progpow::v_0_9_3::COUNT_CACHE, cacheOpCount(buggy));
    EXPECT_EQ(algo::progpow::v_0_9_3::COUNT_MATH, mathOpCount(buggy));
    EXPECT_NE(canonical, buggy);
}


// Same inputs must produce byte-identical kernels (the op sequence is a pure
// function of version/period/counts).
TEST_F(ProgpowKernelOpenCLTest, deterministicForSameInputs)
{
    std::string const first{ generatedKernelOpCounts(
        algo::progpow::VERSION::V_0_9_2,
        3u,
        777ull,
        algo::progpow::v_0_9_2::COUNT_CACHE,
        algo::progpow::v_0_9_2::COUNT_MATH) };

    std::string const second{ generatedKernelOpCounts(
        algo::progpow::VERSION::V_0_9_2,
        4u,
        777ull,
        algo::progpow::v_0_9_2::COUNT_CACHE,
        algo::progpow::v_0_9_2::COUNT_MATH) };

    EXPECT_EQ(first, second);
}


// Whole-family guard: every progpow variant must emit a kernel honoring its own
// canonical counts. If a variant's switch arm or count wiring regresses, the op
// count diverges from the expected (cache, math) pair.
TEST_F(ProgpowKernelOpenCLTest, everyVariantHonorsItsCounts)
{
    struct Variant
    {
        algo::progpow::VERSION version;
        uint32_t               countCache;
        uint32_t               countMath;
        char const*            name;
    };

    Variant const variants[]{
        { algo::progpow::VERSION::V_0_9_2, algo::progpow::v_0_9_2::COUNT_CACHE,
          algo::progpow::v_0_9_2::COUNT_MATH, "v_0_9_2" },
        { algo::progpow::VERSION::KAWPOW, algo::kawpow::COUNT_CACHE, algo::kawpow::COUNT_MATH, "kawpow" },
        { algo::progpow::VERSION::MEOWPOW, algo::meowpow::COUNT_CACHE, algo::meowpow::COUNT_MATH, "meowpow" },
        { algo::progpow::VERSION::FIROPOW, algo::firopow::COUNT_CACHE, algo::firopow::COUNT_MATH, "firopow" },
        { algo::progpow::VERSION::EVRPROGPOW, algo::evrprogpow::COUNT_CACHE, algo::evrprogpow::COUNT_MATH,
          "evrprogpow" },
        { algo::progpow::VERSION::PROGPOWQUAI, algo::progpow_quai::COUNT_CACHE, algo::progpow_quai::COUNT_MATH,
          "progpow_quai" },
    };

    uint32_t deviceId{ 10u };
    for (Variant const& v : variants)
    {
        std::string const src{ generatedKernelOpCounts(v.version, deviceId, 4242ull, v.countCache, v.countMath) };
        ++deviceId;

        ASSERT_FALSE(src.empty()) << v.name;
        EXPECT_EQ(v.countCache, cacheOpCount(src)) << v.name << " cache ops";
        EXPECT_EQ(v.countMath, mathOpCount(src)) << v.name << " math ops";
        EXPECT_EQ(static_cast<size_t>(algo::progpow::DAG_LOADS), mergeEntriesCount(src)) << v.name << " merge_entries";
    }
}
