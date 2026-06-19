#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/amd/kheavyhash.hpp>
#include <resolver/tests/amd.hpp>


struct ResolverKHeavyHashAmdTest : public testing::Test
{
    stratum::StratumJobInfo         jobInfo{};
    resolver::tests::Properties     properties{};
    common::mocker::MockerStratum   stratum{};
    resolver::ResolverAmdKHeavyHash resolver{};

    ResolverKHeavyHashAmdTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);
        if (false == resolver::tests::initializeOpenCL(properties))
        {
            logErr() << "fail init opencl";
        }
        resolver.setDevice(&properties.clDevice);
        resolver.setQueue(&properties.clQueue);
        resolver.setContext(&properties.clContext);
    }

    ~ResolverKHeavyHashAmdTest()
    {
        properties.clDevice = nullptr;
        properties.clContext = nullptr;
        properties.clQueue = nullptr;
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.timestamp = 1234567890ull;
        jobInfo.jobIDStr = "kheavyhash-amd-test";
        jobInfo.headerHash = algo::toHash256("00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff");
        // Maximum target: every pow is <= boundary, so any scanned nonce is a hit.
        jobInfo.boundary = algo::toHash256("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    }
};


TEST_F(ResolverKHeavyHashAmdTest, findNonce)
{
    initializeJob(0ull);
    resolver.updateJobId(jobInfo.jobIDStr); // align resolver jobId so the share is not flagged stale

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));

    resolver.setBlocks(128);
    resolver.setThreads(128);

    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    // kHeavyHash submits array params (jobId, nonce-hex).
    EXPECT_FALSE(stratum.paramSubmit.empty());
}
