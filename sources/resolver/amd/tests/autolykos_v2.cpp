#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/autolykos/autolykos.hpp>
#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/amd/autolykos_v2.hpp>
#include <resolver/tests/amd.hpp>


// Test-only subclass that runs the real dag/search/verify kernels but exposes the
// RAW GPU result buffer -- i.e. the nonces the verify kernel itself accepts against
// the boundary -- before getResultCache() re-filters them through the (divergent)
// CPU mhssamadani::isValidShare. This lets a test assert what the GPU actually does.
struct ProbeResolverAmdAutolykosV2 : public resolver::ResolverAmdAutolykosV2
{
    // Launches search + verify over a SINGLE work-group, so only nonces
    // [base, base + AMD_BLOCK_DIM) are checked. With a real (tight) boundary this
    // makes a single known-good nonce at tid 0 the only solution captured in the
    // 4-slot result buffer, instead of being lost among the many solutions a full
    // NONCES_PER_ITER sweep would find.
    //
    // The kernel argument order MUST mirror ResolverAmdAutolykosV2::executeSync(); if
    // that launch sequence changes, update this in lockstep or the test silently
    // exercises a stale path. (We can't reuse executeSync directly because it also
    // runs getResultCache(), which filters the raw nonces through isValidShare.)
    bool runSingleGroupRaw(algo::autolykos_v2::Result& out)
    {
        uint32_t const          blockDim{ algo::autolykos_v2::AMD_BLOCK_DIM };
        cl::CommandQueue* const queue{ clQueue[currentIndexStream] };

        if (false == parameters.resultCache.resetBufferHost(queue))
        {
            return false;
        }

        cl_int err{ CL_SUCCESS };
        auto&  clKernelSearch{ kernelGeneratorSearch.clKernel };
        err |= clKernelSearch.setArg(0u, *(parameters.headerCache.getBuffer()));
        err |= clKernelSearch.setArg(1u, *(parameters.dagCache.getBuffer()));
        err |= clKernelSearch.setArg(2u, *(parameters.BHashes.getBuffer()));
        err |= clKernelSearch.setArg(3u, parameters.hostNonce);
        err |= clKernelSearch.setArg(4u, parameters.hostPeriod);
        err |= queue->enqueueNDRangeKernel(
            clKernelSearch,
            cl::NullRange,
            cl::NDRange(blockDim, 1, 1),
            cl::NDRange(blockDim, 1, 1));
        err |= queue->finish();
        if (CL_SUCCESS != err)
        {
            return false;
        }

        auto& clKernelVerify{ kernelGeneratorVerify.clKernel };
        err |= clKernelVerify.setArg(0u, *(parameters.boundaryCache.getBuffer()));
        err |= clKernelVerify.setArg(1u, *(parameters.dagCache.getBuffer()));
        err |= clKernelVerify.setArg(2u, *(parameters.BHashes.getBuffer()));
        err |= clKernelVerify.setArg(3u, *(parameters.resultCache.getBuffer()));
        err |= clKernelVerify.setArg(4u, parameters.hostNonce);
        err |= clKernelVerify.setArg(5u, parameters.hostPeriod);
        err |= clKernelVerify.setArg(6u, parameters.hostHeight);
        err |= queue->enqueueNDRangeKernel(
            clKernelVerify,
            cl::NullRange,
            cl::NDRange(blockDim, 1, 1),
            cl::NDRange(blockDim, 1, 1));
        err |= queue->finish();
        if (CL_SUCCESS != err)
        {
            return false;
        }

        return parameters.resultCache.getBufferHost(queue, &out);
    }

    // The header isValidShare re-checks against; must track jobInfo.headerHash.
    algo::hash256 const& hostHeader() const
    {
        return parameters.hostHeader;
    }

    // Copy one 32-byte DAG element (HOST_NO_ACCESS device buffer) out via a small
    // host-readable staging buffer, so a test can compare it to the canonical
    // genElementV2 value and detect indices the fillDAG launch failed to write.
    bool readDagElement(uint64_t const idx, uint8_t* const out)
    {
        cl::Buffer staging(*clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, algo::LEN_HASH_256);
        if (CL_SUCCESS
            != clQueue[currentIndexStream]->enqueueCopyBuffer(
                *(parameters.dagCache.getBuffer()),
                staging,
                idx * algo::LEN_HASH_256,
                0u,
                algo::LEN_HASH_256))
        {
            return false;
        }
        if (CL_SUCCESS != clQueue[currentIndexStream]->enqueueReadBuffer(staging, CL_TRUE, 0u, algo::LEN_HASH_256, out))
        {
            return false;
        }
        return true;
    }
};


struct ResolverAutolykosv2AmdTest : public testing::Test
{
    stratum::StratumJobInfo       jobInfo{};
    resolver::tests::Properties   properties{};
    common::mocker::MockerStratum stratum{};
    ProbeResolverAmdAutolykosV2   resolver{};

    ResolverAutolykosv2AmdTest()
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

    ~ResolverAutolykosv2AmdTest()
    {
        properties.clDevice = nullptr;
        properties.clContext = nullptr;
        properties.clQueue = nullptr;
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce = nonce;
        jobInfo.headerHash = algo::toHash256("d6ff40d44bb470fb3c43b02a67ca3534ff884e2be88484fa89e8c904c0d44392");
        jobInfo.boundary = algo::toHash256("28948096409218832353798863888813816354483909556597628510643976122896");
        jobInfo.blockNumber = 1034782;
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.period = castU64(algo::autolykos_v2::computePeriod(castU32(jobInfo.blockNumber)));
    }
};


TEST_F(ResolverAutolykosv2AmdTest, period)
{
    EXPECT_EQ(104107290u, algo::autolykos_v2::computePeriod(1028992u));
}


TEST_F(ResolverAutolykosv2AmdTest, findNonce)
{
    initializeJob(0x5a710000783f4470);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));

    // updateConstants must propagate the job header to the CPU re-check
    // (isValidShare reads parameters.hostHeader); otherwise every valid GPU share
    // is hashed against a stale header and dropped before submit.
    EXPECT_TRUE(algo::isEqual(resolver.hostHeader(), jobInfo.headerHash));

    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    EXPECT_FALSE(stratum.paramSubmit.empty());
}


// Canonical Autolykos2 proof-of-work vector from the Ergo reference client
// (AutolykosPowSchemeSpec, "test vectors for first increase in N value", height
// 614,400, protocol version 2; algorithm sigma/pow/Autolykos2PowValidation.scala).
// Re-derived independently against Blake2b256 to confirm:
//   msg    = 548c3e60..7e864f  (Blake2b256 of the header without PoW)
//   nonce  = 0x0000000000003105
//   height = 614400 -> N = calcN = 70464240
//   hit    = 0002fcb113fe65e5754959872dfdbffea0489bf830beb4961ddc0e9e66a1412a
//   b      = 7067388259...301849  (target at difficulty 16384)
// hit < b, so nonce 0x3105 is a valid solution. With base nonce 0x3105 the search
// kernel hashes exactly this nonce at tid 0, so a correct GPU pipeline MUST accept
// it. This pins the dag/search/verify kernels to the canonical algorithm,
// independent of the live stratum header/nonce/boundary plumbing.
TEST_F(ResolverAutolykosv2AmdTest, acceptsCanonicalErgoVectorHeight614400)
{
    // N for the vector must match the reference (calcN(614400) == 70464240).
    ASSERT_EQ(70464240u, algo::autolykos_v2::computePeriod(614400u));

    jobInfo.headerHash = algo::toHash256("548c3e602a8f36f8f2738f5f643b02425038044d98543a51cabaa9785e7e864f");
    jobInfo.blockNumber = 614400;
    jobInfo.period = castU64(algo::autolykos_v2::computePeriod(614400u));
    jobInfo.nonce = 0x3105ull; // tid 0 hashes nonce 0x0000000000003105

    // Build the boundary through the exact production pipeline (mirrors
    // StratumAutolykosV2::onMiningNotify) so the verify kernel sees the target in
    // the same little-endian layout it does live.
    jobInfo.boundary = algo::toHash2<algo::hash256, algo::hash512>(
        algo::toLittleEndian<algo::hash512>(algo::decimalToHash<algo::hash512>(
            "7067388259113537318333190002971674063283542741642755394446115914399301849")));
    jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));

    algo::autolykos_v2::Result raw{};
    ASSERT_TRUE(resolver.runSingleGroupRaw(raw));

    EXPECT_TRUE(raw.found);

    bool canonicalFound{ false };
    for (uint32_t i{ 0u }; i < raw.count && i < algo::autolykos_v2::MAX_RESULT; ++i)
    {
        if (0x3105ull == raw.nonces[i])
        {
            canonicalFound = true;
        }
    }
    EXPECT_TRUE(canonicalFound);
}


// Verify the DAG is fully generated at a live mainnet height. At block 1803848 the
// table is N = 216,430,305 elements = ~6.9 GB. Each element must equal the canonical
// genElementV2(idx, height) = Blake2b256(idx_4BE | height_4BE | M)[1..31], which the
// dag kernel stores as the 32-byte hash byte-reversed with the last byte zeroed.
// Expected values below were computed independently (Bun blake2b256). A low index is
// always written; the cross-4GB / end indices expose whether the fillDAG launch (or
// the single >4GB buffer's 32-bit addressing) drops the high part of the table --
// the cause of the live GPU "Share above target" false positives.
TEST_F(ResolverAutolykosv2AmdTest, dagFullyGeneratedAtLiveHeight1803848)
{
    ASSERT_EQ(216430305u, algo::autolykos_v2::computePeriod(1803848u));

    jobInfo.headerHash = algo::toHash256("5128c31fac60b942c1a9ce1441017ec4b7a54d71bf6f88281bb37a3aa1ac23f8");
    jobInfo.blockNumber = 1803848;
    jobInfo.period = castU64(algo::autolykos_v2::computePeriod(1803848u));
    jobInfo.nonce = 0ull;
    jobInfo.boundary = algo::toHash256("ff"); // unused for DAG content
    jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));

    auto const elemHex{ [&](uint64_t const idx) -> std::string
                        {
                            algo::hash256 e{};
                            EXPECT_TRUE(resolver.readDagElement(idx, e.ubytes));
                            return algo::toHex(e);
                        } };

    // Low index -- written by any launch; sanity that DAG build works at all.
    EXPECT_EQ("7d5ed6ebf4dffa8217b77244abecefe1aaa95f16136d7747a555ffbca7d84c00", elemHex(1000ull));
    // Straddle the 4 GiB element boundary (134,217,728 = 4 GiB / 32).
    EXPECT_EQ("2b5bde4ff95f66665f1c0fe01af5f69e27c1456a25ba54abb9837b80c4bc9900", elemHex(134217727ull));
    EXPECT_EQ("a24366848eef17763bd09ec7384737abb3c0f887739eac2096d3fb2713ae1000", elemHex(134217728ull));
    EXPECT_EQ("346822e39669c7307ece71ea84c804b25b31153e9733549035e5cc2304f2e700", elemHex(134218728ull));
    // Last element of the table.
    EXPECT_EQ("58006852cbe55c358518e6ca9f5f21e20f087335bcbc88280bc3761ec3bae400", elemHex(216430304ull));
}
