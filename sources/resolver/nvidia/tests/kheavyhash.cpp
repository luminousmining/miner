#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <common/cast.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/nvidia/kheavyhash.hpp>
#include <resolver/tests/nvidia.hpp>


struct ResolverKHeavyHashNvidiaTest : public testing::Test
{
    stratum::StratumJobInfo            jobInfo{};
    resolver::tests::Properties        properties{};
    common::mocker::MockerStratum      stratum{};
    resolver::ResolverNvidiaKHeavyHash resolver{};

    ResolverKHeavyHashNvidiaTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);
        if (false == resolver::tests::initializeCuda(properties))
        {
            logErr() << "Fail init cuda";
        }
        resolver.cuStream[0] = properties.cuStream;
        resolver.cuProperties = &properties.cuProperties;
        resolver.cuDevice = &properties.cuDevice;
    }

    ~ResolverKHeavyHashNvidiaTest()
    {
        resolver::tests::cleanUpCuda(properties);
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.timestamp = 1234567890ull;
        jobInfo.jobIDStr = "kheavyhash-nvidia-test";
        jobInfo.headerHash = algo::toHash256("00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff");
        // Maximum target: every pow is <= boundary, so any scanned nonce is a hit.
        jobInfo.boundary = algo::toHash256("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    }
};


TEST_F(ResolverKHeavyHashNvidiaTest, findNonce)
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
