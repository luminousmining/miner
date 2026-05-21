#if defined(CUDA_ENABLE)

#include <gtest/gtest.h>

#include <algo/algo_type.hpp>
#include <algo/hash_utils.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/nvidia/random_x.hpp>
#include <resolver/tests/nvidia.hpp>


TEST(ResolverNvidiaRandomXTest, algorithmType)
{
    resolver::ResolverNvidiaRandomX resolver{};
    EXPECT_EQ(algo::ALGORITHM::RANDOM_X, resolver.algorithm);
}


// Full pipeline test: Argon2d cache + SuperscalarHash dataset + VM search.
// WARNING: this test is slow (~1-3 min) due to the Argon2d cache construction.
struct ResolverNvidiaRandomXFullTest : public testing::Test
{
    stratum::StratumJobInfo           jobInfo{};
    resolver::tests::Properties       properties{};
    common::mocker::MockerStratum     stratum{};
    resolver::ResolverNvidiaRandomX   resolver{};

    ResolverNvidiaRandomXFullTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);

        if (false == resolver::tests::initializeCuda(properties))
        {
            logErr() << "Fail init cuda";
        }

        resolver.cuStream[0]    = properties.cuStream;
        resolver.cuProperties   = &properties.cuProperties;
        resolver.cuDevice       = &properties.cuDevice;

        // Enough threads to cover nonce 0 — target is trivially easy so any nonce matches
        resolver.setBlocks(4u);
        resolver.setThreads(64u);
    }

    ~ResolverNvidiaRandomXFullTest()
    {
        resolver::tests::cleanUpCuda(properties);
    }

    void initializeJob()
    {
        // Key: "test key 000" (12 bytes) zero-padded to 32 bytes
        jobInfo.seedHash = algo::hash256{};
        static constexpr uint8_t KEY[]{ 't', 'e', 's', 't', ' ', 'k', 'e', 'y', ' ', '0', '0', '0' };
        for (uint32_t i{ 0u }; i < 12u; ++i)
        {
            jobInfo.seedHash.ubytes[i] = KEY[i];
        }

        // Blob: "test input 000" (14 bytes) zero-padded to 77 bytes
        // Nonce field is at offset 39 and stays 0x00000000 (from zero-padding)
        jobInfo.headerBlob = algo::hash3072{};
        static constexpr uint8_t INPUT[]{ 't', 'e', 's', 't', ' ', 'i', 'n', 'p', 'u', 't', ' ', '0', '0', '0' };
        for (uint32_t i{ 0u }; i < 14u; ++i)
        {
            jobInfo.headerBlob.ubytes[i] = INPUT[i];
        }

        // Trivially easy target: any hash whose last 4 bytes < 0xFFFFFFFF wins
        jobInfo.targetBits   = 0xFFFFFFFFu;
        jobInfo.boundary     = algo::hash256{};
        jobInfo.boundaryU64  = 0xFFFFFFFFull;
        jobInfo.nonce        = 0ull;
    }
};


TEST_F(ResolverNvidiaRandomXFullTest, findNonce)
{
    initializeJob();

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));

    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    // paramSubmit[0] = jobId, [1] = nonce (8 hex chars), [2] = hash (64 hex chars)
    std::string const nonceStr{ stratum.paramSubmit[1].as_string().c_str() };
    std::string const hashStr { stratum.paramSubmit[2].as_string().c_str() };

    // Format check: nonce = 8 hex chars, hash = 64 hex chars
    EXPECT_EQ(8u,  nonceStr.size());
    EXPECT_EQ(64u, hashStr.size());

    // Hash must be non-zero (a real 32-byte hash is astronomically unlikely to be all zeros)
    bool hashNonZero{ false };
    for (uint32_t i{ 0u }; i < 64u; ++i)
    {
        if (hashStr[i] != '0') { hashNonZero = true; break; }
    }
    EXPECT_TRUE(hashNonZero);
}


TEST_F(ResolverNvidiaRandomXFullTest, datasetRebuildOnSeedChange)
{
    initializeJob();
    ASSERT_TRUE(resolver.updateMemory(jobInfo));

    // Second call with the same seed must return true immediately (no rebuild)
    ASSERT_TRUE(resolver.updateMemory(jobInfo));

    // Third call with a different seed forces a rebuild
    jobInfo.seedHash.ubytes[0] ^= 0xFFu;
    ASSERT_TRUE(resolver.updateMemory(jobInfo));
}


// WARNING: slow (~3-5 min) — builds Argon2d cache + SuperscalarHash dataset for real Monero block.
// Validates exact hash against the tevador/RandomX reference implementation.
TEST_F(ResolverNvidiaRandomXFullTest, exactHashBlock3300000)
{
    // Monero block 3300000
    // seed_height = (3300000 - 64 - 1) / 2048 * 2048 = 3299328
    // seed_hash = hash of block 3299328
    static constexpr uint8_t SEED_HASH[32u]
    {
        0xed, 0xe5, 0xe4, 0x39, 0x52, 0x8e, 0xee, 0x83,
        0x66, 0xe0, 0x69, 0x71, 0xe8, 0x81, 0x22, 0x1d,
        0x3f, 0xf1, 0x2a, 0x0b, 0x76, 0x15, 0x2e, 0x4c,
        0x46, 0x2c, 0xa0, 0x64, 0x2d, 0xce, 0x73, 0x2d
    };

    // block_hashing_blob (77 bytes); nonce at offset 39: 54 09 04 92 = LE 0x92040954
    static constexpr uint8_t BLOB[77u]
    {
        0x10, 0x10, 0xb6, 0xdc, 0xe1, 0xba, 0x06, 0x27,
        0x54, 0x86, 0x25, 0x5c, 0x2e, 0x0d, 0xf7, 0x74,
        0x31, 0x63, 0x74, 0xb5, 0x11, 0x4d, 0x16, 0x9d,
        0x4e, 0x97, 0x14, 0xd6, 0x9b, 0xf0, 0x2c, 0x17,
        0xc8, 0x3f, 0x3b, 0x3f, 0x02, 0x88, 0xba,
        0x54, 0x09, 0x04, 0x92,
        0x02, 0xdc, 0xb5, 0xc9, 0x01, 0x01, 0xff, 0xa0,
        0xb5, 0xc9, 0x01, 0x01, 0x80, 0xd9, 0xe5, 0xe5,
        0xd6, 0x11, 0x03, 0x4f, 0x65, 0xbf, 0x7a, 0xe8,
        0xf9, 0x7e, 0x99, 0x11, 0x5b, 0x1c, 0x58, 0x2b,
        0x78, 0x93
    };

    // 1 block × 1 thread: only nonce 0x92040954 is tested
    resolver.setBlocks(1u);
    resolver.setThreads(1u);

    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        jobInfo.seedHash.ubytes[i] = SEED_HASH[i];
    }
    for (uint32_t i{ 0u }; i < 77u; ++i)
    {
        jobInfo.headerBlob.ubytes[i] = BLOB[i];
    }

    jobInfo.targetBits  = 0xFFFFFFFFu;
    jobInfo.boundaryU64 = 0xFFFFFFFFull;
    jobInfo.nonce       = 0x92040954ull;

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));

    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr{ stratum.paramSubmit[1].as_string().c_str() };
    std::string const hashStr { stratum.paramSubmit[2].as_string().c_str() };

    EXPECT_EQ("92040954", nonceStr);
    // Expected hash from tevador/RandomX reference implementation
    EXPECT_EQ("8d19b16a372003be68a62b773d1b282b3b36eda60295f6ed8de31509551884fb", hashStr);
}

#endif
