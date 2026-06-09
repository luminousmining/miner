#include <array>
#include <sstream>

#include <CL/opencl.hpp>

#include <algo/kheavyhash/matrix.hpp>
#include <algo/kheavyhash/types.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/error/opencl_error.hpp>
#include <common/log/log.hpp>
#include <resolver/amd/kheavyhash.hpp>


resolver::ResolverAmdKHeavyHash::ResolverAmdKHeavyHash() : resolver::ResolverAmd()
{
    if (algorithm == algo::ALGORITHM::UNKNOWN)
    {
        algorithm = algo::ALGORITHM::KHEAVYHASH;
    }
}


resolver::ResolverAmdKHeavyHash::~ResolverAmdKHeavyHash()
{
    parameters.matrixCache.free();
    parameters.headerCache.free();
    parameters.targetCache.free();
    parameters.resultCache.free();
}


bool resolver::ResolverAmdKHeavyHash::updateMemory(stratum::StratumJobInfo const& jobInfo)
{
    (void)jobInfo;

    if (nullptr == clContext) [[unlikely]]
    {
        return false;
    }
    if (nullptr == clQueue[0] || nullptr == clQueue[1]) [[unlikely]]
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // No DAG: just (re)allocate the fixed-size per-job buffers and build the
    // search kernel once. The matrix/header/target contents are uploaded in
    // updateConstants (they change every job).
    parameters.matrixCache.free();
    parameters.headerCache.free();
    parameters.targetCache.free();
    parameters.resultCache.free();

    if (false == parameters.matrixCache.alloc(*clContext)
        || false == parameters.headerCache.alloc(clQueue[currentIndexStream], *clContext)
        || false == parameters.targetCache.alloc(clQueue[currentIndexStream], *clContext)
        || false == parameters.resultCache.alloc(clQueue[currentIndexStream], *clContext))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == buildSearch())
    {
        return false;
    }

    return true;
}


bool resolver::ResolverAmdKHeavyHash::updateConstants(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // Host-side matrix generation from the pre-pow header (xoshiro256++ +
    // full-rank gate). This is the CPU reference that the kernel is gated
    // bit-identical against; it never runs on the GPU.
    ::kheavyhash::Hash256 seed{};
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        seed[i] = jobInfo.headerHash.ubytes[i];
    }
    ::kheavyhash::Matrix const matrix{ ::kheavyhash::generateMatrix(seed) };

    std::array<uint16_t, 64u * 64u> flat{};
    for (uint32_t r{ 0u }; r < 64u; ++r)
    {
        for (uint32_t c{ 0u }; c < 64u; ++c)
        {
            flat[r * 64u + c] = matrix[r][c];
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false
        == parameters.matrixCache.write(flat.data(), flat.size() * sizeof(uint16_t), clQueue[currentIndexStream]))
    {
        return false;
    }
    if (false == parameters.headerCache.setBufferDevice(clQueue[currentIndexStream], &jobInfo.headerHash))
    {
        return false;
    }
    if (false == parameters.targetCache.setBufferDevice(clQueue[currentIndexStream], &jobInfo.boundary))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    overrideOccupancy(256u, 8192u);

    return true;
}


bool resolver::ResolverAmdKHeavyHash::buildSearch()
{
    ////////////////////////////////////////////////////////////////////////////
    kernelGenerator.clear();
    // kHeavyHash_lm4: LDS-staged matrix + v_dot4_u32_u8 matmul + powHash keccak
    // midstate (per-job round-1 hoisted to LDS). Bit-identical to the reference
    // `kHeavyHash_lm0` (OpenCL KAT-gated) and measured ~1.46x faster on the RX 9070 XT
    // (gfx1201): ~393 -> ~573 MH/s.
    kernelGenerator.setKernelName("kHeavyHash_lm4");
    kernelGenerator.addDefine("MAX_RESULT", algo::kheavyhash::MAX_RESULT);

    ////////////////////////////////////////////////////////////////////////////
    if (false == kernelGenerator.appendFile("kernel/kheavyhash/kheavyhash.cl"))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == kernelGenerator.build(clDevice, clContext))
    {
        return false;
    }

    return true;
}


bool resolver::ResolverAmdKHeavyHash::executeSync(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    auto& clKernel{ kernelGenerator.clKernel };
    OPENCL_ER(clKernel.setArg(0u, *(parameters.matrixCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(1u, *(parameters.headerCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(2u, *(parameters.targetCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(3u, jobInfo.timestamp));
    OPENCL_ER(clKernel.setArg(4u, jobInfo.nonce));
    OPENCL_ER(clKernel.setArg(5u, *(parameters.resultCache.getBuffer())));

    ////////////////////////////////////////////////////////////////////////////
    size_t const globalSize{ static_cast<size_t>(getBlocks()) * static_cast<size_t>(getThreads()) };
    size_t const localSize{ static_cast<size_t>(getThreads()) };
    OPENCL_ER(clQueue[currentIndexStream]->enqueueNDRangeKernel(
        clKernel,
        cl::NullRange,
        cl::NDRange(globalSize),
        cl::NDRange(localSize)));
    OPENCL_ER(clQueue[currentIndexStream]->finish());

    ////////////////////////////////////////////////////////////////////////////
    if (false == getResultCache(jobInfo.jobIDStr))
    {
        return false;
    }

    return true;
}


bool resolver::ResolverAmdKHeavyHash::executeAsync(stratum::StratumJobInfo const& jobInfo)
{
    return executeSync(jobInfo);
}


bool resolver::ResolverAmdKHeavyHash::getResultCache(std::string const& _jobId)
{
    algo::kheavyhash::Result data{};

    if (false == parameters.resultCache.getBufferHost(clQueue[currentIndexStream], &data))
    {
        return false;
    }

    if (true == data.found)
    {
        uint32_t const count{ common::max_limit(data.count, algo::kheavyhash::MAX_RESULT) };

        resultShare.found = true;
        resultShare.count = count;
        resultShare.extraNonceSize = 0u;
        resultShare.jobId.assign(_jobId);

        for (uint32_t i{ 0u }; i < count; ++i)
        {
            resultShare.nonces[i] = data.nonces[i];
        }

        if (false == parameters.resultCache.resetBufferHost(clQueue[currentIndexStream]))
        {
            return false;
        }
    }

    return true;
}


void resolver::ResolverAmdKHeavyHash::submit(stratum::Stratum* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i{ 0u }; i < resultShare.count; ++i)
            {
                std::stringstream nonceHexa;
                nonceHexa << std::hex << resultShare.nonces[i];

                boost::json::array params{ resultShare.jobId, nonceHexa.str() };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }

        resultShare.count = 0u;
        resultShare.found = false;
    }
}


void resolver::ResolverAmdKHeavyHash::submit(stratum::StratumSmartMining* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i{ 0u }; i < resultShare.count; ++i)
            {
                std::stringstream nonceHexa;
                nonceHexa << std::hex << resultShare.nonces[i];

                boost::json::array params{ resultShare.jobId, nonceHexa.str() };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }

        resultShare.count = 0u;
        resultShare.found = false;
    }
}
