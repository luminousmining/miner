#include <CL/opencl.hpp>

#include <algo/bitwise.hpp>
#include <algo/keccak.hpp>
#include <algo/autolykos/autolykos.hpp>
#include <common/cast.hpp>
#include <common/chrono.hpp>
#include <common/custom.hpp>
#include <common/error/opencl_error.hpp>
#include <common/log/log.hpp>
#include <resolver/amd/autolykos_v2.hpp>


resolver::ResolverAmdAutolykosV2::~ResolverAmdAutolykosV2()
{
    parameters.BHashes.free();
    parameters.dagCache.free();
    parameters.boundaryCache.free();
    parameters.headerCache.free();
    parameters.resultCache.free();
}


bool resolver::ResolverAmdAutolykosV2::updateMemory(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    if (nullptr == clContext || nullptr == clQueue)
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    parameters.hostPeriod = castU32(jobInfo.period);
    parameters.hostHeight = algo::be::U32(castU32(jobInfo.blockNumber));
    parameters.hostDagItemCount = castU32(jobInfo.period);

    ////////////////////////////////////////////////////////////////////////////
    parameters.BHashes.free();
    parameters.dagCache.free();
    parameters.boundaryCache.free();
    parameters.headerCache.free();
    parameters.resultCache.free();

    ////////////////////////////////////////////////////////////////////////////
    parameters.BHashes.setCapacity(algo::autolykos_v2::NONCES_PER_ITER);
    parameters.dagCache.setCapacity(parameters.hostDagItemCount);

    ////////////////////////////////////////////////////////////////////////////
    if (   false == parameters.BHashes.alloc(*clContext)
        || false == parameters.dagCache.alloc(*clContext)
        || false == parameters.boundaryCache.alloc(clQueue, *clContext)
        || false == parameters.headerCache.alloc(clQueue, *clContext)
        || false == parameters.resultCache.alloc(clQueue, *clContext))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == kernelGeneratorDAG.isBuilt())
    {
        if (false == buildDAG())
        {
            return false;
        }
    }

    if (   false == kernelGeneratorSearch.isBuilt()
        || false == kernelGeneratorVerify.isBuilt())
    {
        if (false == buildSearch())
        {
            return false;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == fillDAG())
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverAmdAutolykosV2::updateConstants(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    parameters.hostNonce = jobInfo.nonce;
    parameters.hostPeriod = castU32(jobInfo.period);
    parameters.hostHeight = algo::be::U32(castU32(jobInfo.blockNumber));
    parameters.hostDagItemCount = castU32(jobInfo.period);

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const* const boundary { jobInfo.boundary.word32 };
    if (false == parameters.boundaryCache.setBufferDevice(clQueue, boundary))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const* const header { jobInfo.headerHash.word32 };
    if (false == parameters.headerCache.setBufferDevice(clQueue, header))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    setBlocks(1u);
    setThreads(algo::autolykos_v2::AMD_NONCES_PER_ITER);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverAmdAutolykosV2::buildDAG()
{
    ////////////////////////////////////////////////////////////////////////////
    kernelGeneratorDAG.clear();

    ////////////////////////////////////////////////////////////////////////////
    kernelGeneratorDAG.setKernelName("autolykos_v2_build_dag");

    ////////////////////////////////////////////////////////////////////////////
    if (   false == kernelGeneratorDAG.appendFile("kernel/common/rotate_byte.cl")
        || false == kernelGeneratorDAG.appendFile("kernel/crypto/blake2b_compress.cl")
        || false == kernelGeneratorDAG.appendFile("kernel/autolykos/autolykos_v2_dag.cl"))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == kernelGeneratorDAG.buildOpenCL(clDevice, clContext))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverAmdAutolykosV2::buildSearch()
{
    if (   false == buildKernelSearch()
        || false == buildKernelVerify())
    {
        return false;
    }

    return true;
}


bool resolver::ResolverAmdAutolykosV2::fillDAG()
{
    ////////////////////////////////////////////////////////////////////////////
    auto& clKernel { kernelGeneratorDAG.clKernel };
    OPENCL_ER(clKernel.setArg(0u, *(parameters.dagCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(1u, parameters.hostHeight));
    OPENCL_ER(clKernel.setArg(2u, parameters.hostPeriod));
    uint32_t const blockDim { algo::autolykos_v2::AMD_BLOCK_DIM };
    uint32_t globalDimX { ((parameters.hostPeriod / blockDim) + 1) * blockDim };
    OPENCL_ER(
        clQueue->enqueueNDRangeKernel(
            clKernel,
            cl::NullRange,
            cl::NDRange(globalDimX, 1, 1),
            cl::NDRange(blockDim,   1, 1)));
    OPENCL_ER(clQueue->finish());

    return true;
}


bool resolver::ResolverAmdAutolykosV2::buildKernelSearch()
{
    ////////////////////////////////////////////////////////////////////////////
    kernelGeneratorSearch.clear();

    ////////////////////////////////////////////////////////////////////////////
    kernelGeneratorSearch.setKernelName("autolykos_v2_search");

    ////////////////////////////////////////////////////////////////////////////
    kernelGeneratorSearch.addDefine("NONCES_PER_ITER", algo::autolykos_v2::AMD_NONCES_PER_ITER);
    kernelGeneratorSearch.addDefine("THREADS_PER_ITER", algo::autolykos_v2::AMD_THREADS_PER_ITER);
    kernelGeneratorSearch.addDefine("K_LEN", algo::autolykos_v2::K_LEN);
    kernelGeneratorSearch.addDefine("NONCE_SIZE_32", algo::autolykos_v2::NONCE_SIZE_32);
    kernelGeneratorSearch.addDefine("NUM_SIZE_32", algo::autolykos_v2::NUM_SIZE_32);

    ////////////////////////////////////////////////////////////////////////////
    if (   false == kernelGeneratorSearch.appendFile("kernel/autolykos/autolykos_v2_result.cl")
        || false == kernelGeneratorSearch.appendFile("kernel/common/rotate_byte.cl")
        || false == kernelGeneratorSearch.appendFile("kernel/crypto/blake2b.cl")
        || false == kernelGeneratorSearch.appendFile("kernel/autolykos/autolykos_v2_var_global.cl")
        || false == kernelGeneratorSearch.appendFile("kernel/autolykos/autolykos_v2_search.cl"))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == kernelGeneratorSearch.buildOpenCL(clDevice, clContext))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverAmdAutolykosV2::buildKernelVerify()
{
    ////////////////////////////////////////////////////////////////////////////
    kernelGeneratorVerify.clear();

    ////////////////////////////////////////////////////////////////////////////
    kernelGeneratorVerify.setKernelName("autolykos_v2_verify");

    ////////////////////////////////////////////////////////////////////////////
    kernelGeneratorVerify.addDefine("NONCES_PER_ITER", algo::autolykos_v2::AMD_NONCES_PER_ITER);
    kernelGeneratorVerify.addDefine("THREADS_PER_ITER", algo::autolykos_v2::AMD_THREADS_PER_ITER);
    kernelGeneratorVerify.addDefine("K_LEN", algo::autolykos_v2::K_LEN);
    kernelGeneratorVerify.addDefine("NONCE_SIZE_32", algo::autolykos_v2::NONCE_SIZE_32);
    kernelGeneratorVerify.addDefine("NUM_SIZE_32", algo::autolykos_v2::NUM_SIZE_32);
    kernelGeneratorVerify.addDefine("NUM_SIZE_8", algo::autolykos_v2::NUM_SIZE_8);

    ////////////////////////////////////////////////////////////////////////////
    if (   false == kernelGeneratorVerify.appendFile("kernel/autolykos/autolykos_v2_result.cl")
        || false == kernelGeneratorVerify.appendFile("kernel/common/rotate_byte.cl")
        || false == kernelGeneratorVerify.appendFile("kernel/crypto/blake2b.cl")
        || false == kernelGeneratorVerify.appendFile("kernel/autolykos/autolykos_v2_var_global.cl")
        || false == kernelGeneratorVerify.appendFile("kernel/autolykos/autolykos_v2_verify.cl"))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == kernelGeneratorVerify.buildOpenCL(clDevice, clContext))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverAmdAutolykosV2::execute(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    parameters.hostNonce = jobInfo.nonce;

    ////////////////////////////////////////////////////////////////////////////
    auto& clKernel { kernelGeneratorSearch.clKernel };
    OPENCL_ER(clKernel.setArg(0u, *(parameters.headerCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(1u, *(parameters.dagCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(2u, *(parameters.BHashes.getBuffer())));
    OPENCL_ER(clKernel.setArg(3u, parameters.hostNonce));
    OPENCL_ER(clKernel.setArg(4u, parameters.hostPeriod));
    OPENCL_ER(
        clQueue->enqueueNDRangeKernel(
            clKernel,
            cl::NullRange,
            cl::NDRange(maxGroupSizeSearch,                1, 1),
            cl::NDRange(algo::autolykos_v2::AMD_BLOCK_DIM, 1, 1)));
    OPENCL_ER(clQueue->finish());

    ////////////////////////////////////////////////////////////////////////////
    auto& clKernelVerify { kernelGeneratorVerify.clKernel };
    OPENCL_ER(clKernelVerify.setArg(0u, *(parameters.boundaryCache.getBuffer())));
    OPENCL_ER(clKernelVerify.setArg(1u, *(parameters.dagCache.getBuffer())));
    OPENCL_ER(clKernelVerify.setArg(2u, *(parameters.BHashes.getBuffer())));
    OPENCL_ER(clKernelVerify.setArg(3u, *(parameters.resultCache.getBuffer())));
    OPENCL_ER(clKernelVerify.setArg(4u, parameters.hostNonce));
    OPENCL_ER(clKernelVerify.setArg(5u, parameters.hostPeriod));
    OPENCL_ER(clKernelVerify.setArg(6u, parameters.hostHeight));
    OPENCL_ER(
        clQueue->enqueueNDRangeKernel(
            clKernelVerify,
            cl::NullRange,
            cl::NDRange(maxGroupSizeVerify,                1, 1),
            cl::NDRange(algo::autolykos_v2::AMD_BLOCK_DIM, 1, 1)));
    OPENCL_ER(clQueue->finish());

    ////////////////////////////////////////////////////////////////////////////
    if (false == getResultCache(jobInfo.jobIDStr,
                                jobInfo.extraNonceSize,
                                jobInfo.extraNonce2Size))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverAmdAutolykosV2::getResultCache(
    std::string const& _jobId,
    uint32_t const extraNonceSize,
    uint32_t const extraNonce2Size)
{
    algo::autolykos_v2::Result data{};

    ////////////////////////////////////////////////////////////////////////////
    if (false == parameters.resultCache.getBufferHost(clQueue, &data))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == data.found)
    {
        uint32_t const count
        {
            MAX_LIMIT(data.count, algo::autolykos_v2::MAX_RESULT)
        };

        resultShare.found = true;
        resultShare.count = count;
        resultShare.extraNonceSize = extraNonceSize;
        resultShare.extraNonce2Size = extraNonce2Size;
        resultShare.jobId.assign(_jobId);

        for (uint32_t i { 0u }; i < count; ++i)
        {
            resultShare.nonces[i] = data.nonces[i];
        }

        if (false == parameters.resultCache.resetBufferHost(clQueue))
        {
            return false;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


void resolver::ResolverAmdAutolykosV2::submit(
    stratum::Stratum* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i { 0u }; i < resultShare.count; ++i)
            {
                std::stringstream nonceHexa;
                nonceHexa << std::hex << resultShare.nonces[i];

                boost::json::array params
                {
                    resultShare.jobId,
                    nonceHexa.str().substr(resultShare.extraNonceSize),
                    nonceHexa.str()
                };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }

        }
    }

    resultShare.count = 0u;
    resultShare.found = false;
}


void resolver::ResolverAmdAutolykosV2::submit(
    stratum::StratumSmartMining* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i { 0u }; i < resultShare.count; ++i)
            {
                std::stringstream nonceHexa;
                nonceHexa << std::hex << resultShare.nonces[i];

                boost::json::array params
                {
                    resultShare.jobId,
                    nonceHexa.str().substr(resultShare.extraNonceSize),
                    nonceHexa.str()
                };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }

        }
    }

    resultShare.count = 0u;
    resultShare.found = false;
}
