#include <algo/cuckatoo/cuckatoo.hpp>
#include <algo/cuckatoo/result.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <resolver/amd/cuckatoo32.hpp>
#include <stratum/smart_mining.hpp>
#include <stratum/stratum.hpp>

#if defined(AMD_ENABLE)


resolver::ResolverAmdCuckatoo32::ResolverAmdCuckatoo32()
{
    overrideOccupancy(algo::cuckatoo::DEFAULT_THREADS, algo::cuckatoo::DEFAULT_BLOCKS);
}


resolver::ResolverAmdCuckatoo32::~ResolverAmdCuckatoo32()
{
    clEdgeBitmap  = cl::Buffer{};
    clNodeCounter = cl::Buffer{};
    clResult      = cl::Buffer{};
}


bool resolver::ResolverAmdCuckatoo32::updateMemory(
    [[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // Allocate GPU buffers (only on first call or after device reset)
    if (false == allocateBuffers())
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Build OpenCL kernels (only if not already compiled)
    if (false == kernelSeed.isBuilt() || false == kernelTrim.isBuilt() || false == kernelCycle.isBuilt())
    {
        if (false == buildKernels())
        {
            return false;
        }
    }

    return true;
}


bool resolver::ResolverAmdCuckatoo32::updateConstants(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // Store job context needed by submit()
    resultShare.jobId     = jobInfo.jobIDStr;
    resultShare.height    = jobInfo.blockNumber;
    resultShare.grinJobId = jobInfo.grinJobId;
    resultShare.nonce     = jobInfo.nonce;
    resultShare.found     = false;

    ////////////////////////////////////////////////////////////////////////////
    // TODO: Upload (pre_pow || nonce) derived SipHash key to GPU.
    //
    // Steps:
    //   1. Decode pre_pow bytes from jobInfo.headerBlob (jobInfo.prePowSize bytes)
    //   2. Append jobInfo.nonce as 8 bytes little-endian
    //   3. Compute H1 = blake2b-256(pre_pow || nonce)
    //   4. Compute H2 = blake2b-256(H1)  — the 32-byte SipHash seed
    //   5. Write H2 into a __constant buffer accessible by all three kernels
    //
    // See: https://github.com/mimblewimble/grin/blob/master/pow/src/cuckatoo/lean.rs

    return true;
}


bool resolver::ResolverAmdCuckatoo32::executeSync(stratum::StratumJobInfo const& jobInfo)
{
    if (false == runTrimming(jobInfo))
    {
        return false;
    }
    if (false == runCycleDetection(jobInfo))
    {
        return false;
    }
    return getResult();
}


bool resolver::ResolverAmdCuckatoo32::executeAsync(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // Cuckatoo32 is too memory-intensive for true double-buffering on most GPUs.
    // Fall back to synchronous execution here; the device loop still advances
    // the nonce between calls, so throughput is maintained.
    return executeSync(jobInfo);
}


void resolver::ResolverAmdCuckatoo32::submit(stratum::Stratum* const stratum)
{
    ////////////////////////////////////////////////////////////////////////////
    if (false == resultShare.found || nullptr == stratum)
    {
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    boost::json::array powArray;
    for (uint32_t i{ 0u }; i < algo::cuckatoo::PROOF_SIZE; ++i)
    {
        powArray.push_back(resultShare.proof[i]);
    }

    boost::json::object params;
    params["height"]  = resultShare.height;
    params["job_id"]  = resultShare.grinJobId;
    params["nonce"]   = resultShare.nonce;
    params["pow"]     = powArray;

    stratum->miningSubmit(deviceId, params);

    resultShare.found = false;
}


void resolver::ResolverAmdCuckatoo32::submit(
    [[maybe_unused]] stratum::StratumSmartMining* const stratum)
{
    // Cuckatoo32 is not supported by the Smart Mining profile.
}


bool resolver::ResolverAmdCuckatoo32::buildKernels()
{
    ////////////////////////////////////////////////////////////////////////////
    // TODO: Implement OpenCL kernel compilation using KernelGeneratorOpenCL.
    //
    // Expected kernel files:
    //   sources/algo/cuckatoo/opencl/cuckatoo32_seed.cl   – edge seeding
    //   sources/algo/cuckatoo/opencl/cuckatoo32_trim.cl   – degree trimming
    //   sources/algo/cuckatoo/opencl/cuckatoo32_cycle.cl  – cycle detection
    //
    // Example (based on ResolverAmdAutolykosV2::buildSearch):
    //   kernelSeed.setKernelName("cuckatoo32_seed");
    //   kernelSeed.addDefine("EDGE_BITS", algo::cuckatoo::EDGE_BITS);
    //   kernelSeed.appendFile("sources/algo/cuckatoo/opencl/cuckatoo32_seed.cl");
    //   return kernelSeed.build(*clDevice, *clContext);

    logErr() << "ResolverAmdCuckatoo32: OpenCL kernels not yet implemented";
    return false;
}


bool resolver::ResolverAmdCuckatoo32::allocateBuffers()
{
    ////////////////////////////////////////////////////////////////////////////
    // TODO: Allocate GPU buffers via clCreateBuffer / cl::Buffer constructor.
    //
    //   clEdgeBitmap  = cl::Buffer(*clContext, CL_MEM_READ_WRITE,
    //                              algo::cuckatoo::EDGE_BITMAP_BYTES);
    //
    //   clNodeCounter = cl::Buffer(*clContext, CL_MEM_READ_WRITE,
    //                              algo::cuckatoo::NUM_NODES);  // 4 GB – 1 byte per node
    //
    //   clResult      = cl::Buffer(*clContext, CL_MEM_READ_WRITE,
    //                              sizeof(algo::cuckatoo::Result));
    //
    // Check deviceMemoryAvailable before allocating; return false if insufficient.

    return false;
}


bool resolver::ResolverAmdCuckatoo32::runTrimming(
    [[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // TODO: Launch kernelSeed then kernelTrim (TRIM_ROUNDS times).
    //
    // Seed phase: for each edge index e in [0, NUM_EDGES):
    //   (u, v) = sipHash24(key, 2*e) & NODE_MASK,  sipHash24(key, 2*e+1) & NODE_MASK
    //   set bit e in clEdgeBitmap
    //
    // Trim phase (repeated TRIM_ROUNDS times):
    //   1. Count node degrees from live edges
    //   2. For each live edge (u,v): if degree[u] < 2 or degree[v] < 2 → clear bit
    //
    // See: https://github.com/tromp/cuckoo for reference C++ implementation

    return false;
}


bool resolver::ResolverAmdCuckatoo32::runCycleDetection(
    [[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // TODO: Launch kernelCycle on the trimmed edge set.
    //
    // The cycle-detection kernel searches for a path of length 42 in the
    // remaining bipartite graph.  On success it writes:
    //   result.found = true
    //   result.proof[0..41] = sorted list of the 42 edge indices forming the cycle
    //
    // The found nonce is the jobInfo.nonce value set in updateConstants().

    return false;
}


bool resolver::ResolverAmdCuckatoo32::getResult()
{
    ////////////////////////////////////////////////////////////////////////////
    // TODO: Read clResult buffer from GPU.
    //
    //   algo::cuckatoo::Result gpuResult{};
    //   clQueue[0]->enqueueReadBuffer(clResult, CL_TRUE, 0,
    //                                  sizeof(gpuResult), &gpuResult);
    //   if (true == gpuResult.found)
    //   {
    //       resultShare.found = true;
    //       std::copy(gpuResult.proof, gpuResult.proof + algo::cuckatoo::PROOF_SIZE,
    //                 resultShare.proof);
    //   }

    return true;
}

#endif
