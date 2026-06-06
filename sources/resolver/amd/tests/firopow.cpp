#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/progpow/firopow.hpp>
#include <common/config.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/amd/firopow.hpp>
#include <resolver/tests/amd.hpp>


struct ResolverFiropowAmdTest : public testing::Test
{
    stratum::StratumJobInfo       jobInfo{};
    resolver::tests::Properties   properties{};
    resolver::ResolverAmdFiroPOW  resolver{};
    common::mocker::MockerStratum stratum{};

    ResolverFiropowAmdTest()
    {
        ////////////////////////////////////////////////////////////////////////////
        common::Config& config{ common::Config::instance() };
        config.deviceAlgorithm.ethashBuildLightCacheCPU = true;

        ////////////////////////////////////////////////////////////////////////////
        common::setLogLevel(common::TYPELOG::__DEBUG);

        if (false == resolver::tests::initializeOpenCL(properties))
        {
            logErr() << "fail init opencl";
        }

        resolver.setDevice(&properties.clDevice);
        resolver.setQueue(&properties.clQueue);
        resolver.setContext(&properties.clContext);
    }

    ~ResolverFiropowAmdTest()
    {
        properties.clDevice = nullptr;
        properties.clContext = nullptr;
        properties.clQueue = nullptr;
    }

    // Authoritative FiroPoW vector (firoorg/firo firopow_test_vectors.hpp),
    // block 1 / epoch 0. Low epoch keeps the DAG small and the hash correct
    // (see the DISABLED epoch-419 test for the known large-DAG defect).
    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.blockNumber = 1ull;
        jobInfo.headerHash = algo::toHash256("2d794e900dcad779e658de9078d9a88eee87d75f7b09a8fdd270d3a8e76650c7");
        jobInfo.boundary = algo::toHash256("0001869e7a058e2aaf266cd2f166fb85c98d651e60eadbbe72bb0a36f8802805");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.epoch = 0;
        jobInfo.period = jobInfo.blockNumber / algo::firopow::MAX_PERIOD;
    }
};


TEST_F(ResolverFiropowAmdTest, findNonce)
{
    initializeJob(0x85f22c9b3cd2f123);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    // One workgroup starting at the vector nonce -> deterministic single result.
    resolver.setThreads(1u);
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr{ stratum.paramSubmit[1].as_string().c_str() };

    using namespace std::string_literals;
    EXPECT_EQ("0x85f22c9b3cd2f123"s, nonceStr);
}


TEST_F(ResolverFiropowAmdTest, notFindNonce)
{
    initializeJob(0x85f22c9b3cd2f123);
    jobInfo.boundaryU64 = 1ull;  // unsatisfiable boundary -> no share can clear it

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    resolver.setThreads(1u);
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmit.empty());
}


// Verifies the *submitted mixhash* against authoritative FiroPoW test vectors
// (firoorg/firo: src/crypto/progpow/firopow_test_vectors.hpp). The existing tests
// only check that the nonce is found; a wrong-but-self-consistent digest would
// still pass them yet be rejected live as "Invalid Firo Mixhash". These pin the
// digest itself, across epochs (epoch 0 does not exercise DAG growth/seed).
namespace
{
    void checkFiroVector(
        ResolverFiropowAmdTest& t,
        uint64_t const          nonce,
        uint64_t const          blockNumber,
        uint32_t const          epoch,
        char const* const       headerHashHex,
        char const* const       boundaryHex,
        char const* const       expectedNonce,
        char const* const       expectedMix)
    {
        t.jobInfo.nonce       = nonce;
        t.jobInfo.blockNumber = blockNumber;
        t.jobInfo.headerHash  = algo::toHash256(headerHashHex);
        t.jobInfo.boundary    = algo::toHash256(boundaryHex);
        t.jobInfo.boundaryU64 = algo::toUINT64(t.jobInfo.boundary);
        t.jobInfo.epoch       = static_cast<int32_t>(epoch);
        t.jobInfo.period      = t.jobInfo.blockNumber / algo::firopow::MAX_PERIOD;

        ASSERT_TRUE(t.resolver.updateMemory(t.jobInfo));
        ASSERT_TRUE(t.resolver.updateConstants(t.jobInfo));
        // One workgroup (256 nonces) starting at the vector nonce: the vector nonce
        // is computed and essentially nothing else clears the boundary.
        t.resolver.setThreads(1u);
        ASSERT_TRUE(t.resolver.executeSync(t.jobInfo));
        t.resolver.submit(&t.stratum);

        ASSERT_FALSE(t.stratum.allSubmits.empty()) << "no share cleared boundary (wrong hash?)";

        bool found{ false };
        for (auto const& params : t.stratum.allSubmits)
        {
            if (std::string(expectedNonce) == std::string(params[1].as_string().c_str()))
            {
                found = true;
                EXPECT_EQ(std::string(expectedMix), std::string(params[2].as_string().c_str()));
            }
        }
        EXPECT_TRUE(found) << "vector nonce was not among submitted shares";
    }
}


TEST_F(ResolverFiropowAmdTest, submitMixMatchesFiroVector)  // block 1, epoch 0
{
    checkFiroVector(*this, 0x85f22c9b3cd2f123ull, 1ull, 0,
        "2d794e900dcad779e658de9078d9a88eee87d75f7b09a8fdd270d3a8e76650c7",
        "0001869e7a058e2aaf266cd2f166fb85c98d651e60eadbbe72bb0a36f8802805",
        "0x85f22c9b3cd2f123",
        "0xcfab3766331d6c4e6913e6688a71e4c26b7f36c1581cdbec0f5b19db8956eb50");
}


TEST_F(ResolverFiropowAmdTest, submitMixMatchesFiroVectorEpoch1)  // block 1300, epoch 1
{
    checkFiroVector(*this, 0xdec25420bac29b01ull, 1300ull, 1,
        "1fc36bd5d1bff8d134e24a997cfa43cbb2a0b956379bdc0c8df444f2553f6b7d",
        "00030d3cf40b1c555e4cd9a5e2cdf70b931aca3cc1d5b77ce576146df100500a",
        "0xdec25420bac29b01",
        "0x5f8fe6069efb88d9af861999973b3523295d3b1ec7e8423c965f33b3d12b20b1");
}


TEST_F(ResolverFiropowAmdTest, submitMixMatchesFiroVectorEpoch10)  // block 13000, epoch 10
{
    checkFiroVector(*this, 0x10d0835970ff1254ull, 13000ull, 10,
        "794688e6167995f4891124d488365df76becfd2fc47c5c8337c9a545801d8e60",
        "0001e84617ccaeb80f88c38f5fa4985fb9a2434e6dd31586e7f5a404cd315b1d",
        "0x10d0835970ff1254",
        "0x653b09240717ddc1d251e75cd24b5fb35d7890a08be6e2ac6b97105f0ff5b27d");
}


// block 545860 -> epoch 419, the regime real Firo blocks live in (~5 GB DAG).
// Regression test for the grid-stride DAG-build fix: the AMD gfx1201/RDNA4 driver
// silently clamps the build launch at ~2^24 pages, leaving most of a multi-GB DAG
// zero, so the digest was self-consistent but wrong (epoch 0/1/10 with smaller
// DAGs passed; this and the live Firo epoch failed). Without the fix this fails.
TEST_F(ResolverFiropowAmdTest, submitMixMatchesFiroVectorEpoch419)  // block 545860
{
    checkFiroVector(*this, 0x844c83fd15ddbc98ull, 545860ull, 419,
        "421ecabe666b8a4b3c5d9f899a15115f1fc8e2644468459c9f200d2dd10c204c",
        "00004e1fb2011c6ef320e5a5fc1b76807358965ce1ccf5fcc5dda026cd038963",
        "0x844c83fd15ddbc98",
        "0x19b62e800a00b6ad8208eb53baeebe9d5e96018ccc7df2ebe35cac463984ea4b");
}
