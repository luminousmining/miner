#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/cpu/blake3.hpp>
#include <stratum/job_info.hpp>


namespace
{
    // 302-byte header + KAT nonce + double-BLAKE3 digest, copied verbatim from the
    // independently-verified oracle in sources/algo/blake3/tests/blake3_ref.cpp.
    constexpr char const* HEADER_HEX{
        "000700000000000022d30e3358af8cd1a732e46d47254bb81ffa43cf402dfe001cf5000000000001bd38686272dfd4b55c3559391a"
        "dfab12413c197fa92729465aea000000000001ea959d9fbdd9abeda14a65c0040bb7e626d7c13a6e51979a434f000000000002776a"
        "5133a5941c8bf35e38043a1f4c5700c0844cb835c414c4300000000000017266826c86467877ca053f0776c56a9d340f634237a91e"
        "3865410000000000009dc9c87ce69910b950d0967c9a37930fcfd34f510fc0a014861200000000000157c2d1db2a2af17d3e7560bf"
        "c78b270d2b7e65a7fcff312b3b8310249f0949d3463117e80375771047ab3309102f365ddc609b36eaae1363dfceb25811b3902af4"
        "dd41490d75b7f79a0478f7f721681c59178a2564b84561559a0000018ea81c4bfa1b029ed6"
    };
    constexpr uint64_t    KAT_NONCE{ 0x914544566c9a0a4dull };
    constexpr char const* DIGEST_HEX{ "394696ad2377a8ce8525032656e819183c0585d818ff1cffb52aca6acde2d095" };
}


struct ResolverBlake3CpuTest : public testing::Test
{
    stratum::StratumJobInfo       jobInfo{};
    common::mocker::MockerStratum stratum{};
    resolver::ResolverCpuBlake3   resolver{};

    void initializeJob()
    {
        jobInfo.headerBlob = algo::toHash<algo::hash3072>(HEADER_HEX, algo::HASH_SHIFT::LEFT);
        jobInfo.fromGroup = 1u;
        jobInfo.toGroup = 1u;
        jobInfo.extraNonceSize = 0u;
        jobInfo.jobIDStr.assign("job-1");
        // Device::updateJob calls resolver->updateJobId before submit; mirror it so
        // submit() does not treat the share as stale.
        resolver.updateJobId("job-1");
        // Scan exactly one nonce per execute call: blocks*threads == 1.
        resolver.setBlocks(1u);
        resolver.setThreads(1u);
    }
};


// A trivially-satisfiable target (all 0xFF) means any digest is a winner. Pointing the
// scan base at the KAT nonce proves the scan/record/submit pipeline end to end.
TEST_F(ResolverBlake3CpuTest, findsNonceThenSubmits)
{
    initializeJob();
    for (uint32_t i{ 0u }; i < algo::LEN_HASH_256_WORD_8; ++i)
    {
        jobInfo.targetBlob.ubytes[i] = 0xFFu;
    }
    jobInfo.nonce = KAT_NONCE;

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));

    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmitObject.empty());
    // 8-byte search value (16 hex) zero-padded to the 24-byte (48 hex) submit nonce.
    EXPECT_EQ("914544566c9a0a4d00000000000000000000000000000000",
              std::string{ stratum.paramSubmitObject.at("nonce").as_string().c_str() });
    EXPECT_EQ("job-1", std::string{ stratum.paramSubmitObject.at("jobId").as_string().c_str() });
}


// Exact-equality target: the KAT nonce's digest == target, so memcmp == 0 (<= 0) is a hit,
// proving hashRef is wired correctly (not just any-nonce plumbing).
TEST_F(ResolverBlake3CpuTest, acceptsWhenDigestEqualsTarget)
{
    initializeJob();
    jobInfo.targetBlob = algo::toHash256(DIGEST_HEX);
    jobInfo.nonce = KAT_NONCE;

    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmitObject.empty());
}


// A zero target rejects every real digest (no digest is <= 0). No submit must occur.
TEST_F(ResolverBlake3CpuTest, rejectsWhenAboveTarget)
{
    initializeJob();
    jobInfo.targetBlob = algo::toHash256("0000000000000000000000000000000000000000000000000000000000000000");
    jobInfo.nonce = KAT_NONCE;

    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmitObject.empty());
}
