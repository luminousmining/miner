#include <CL/opencl.hpp>

#include <algo/keccak.hpp>
#include <algo/ethash/ethash.hpp>
#include <common/cast.hpp>
#include <common/chrono.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <common/error/opencl_error.hpp>
#include <resolver/amd/ethash.hpp>


void resolver::ResolverAmdEthash::updateContext(
    stratum::StratumJobInfo const& jobInfo)
{
    algo::ethash::initializeDagContext(context,
                                       jobInfo.epoch,
                                       algo::ethash::MAX_EPOCH_NUMBER,
                                       algo::ethash::DAG_COUNT_ITEMS_GROWTH,
                                       algo::ethash::DAG_COUNT_ITEMS_INIT);
}


bool resolver::ResolverAmdEthash::updateMemory(
    stratum::StratumJobInfo const& jobInfo)
{
    if (nullptr == clContext)
    {
        return false;
    }
    if (nullptr == clQueue)
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    updateContext(jobInfo);

    ////////////////////////////////////////////////////////////////////////////
    SAFE_DELETE(parameters.lightCache);
    SAFE_DELETE(parameters.dagCache);

    ////////////////////////////////////////////////////////////////////////////
    OPENCL_CATCH(
        parameters.lightCache = new (std::nothrow) cl::Buffer(
                *clContext,
                CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                context.lightCache.size));
    OPENCL_CATCH(
        parameters.dagCache = new (std::nothrow) cl::Buffer(
                *clContext,
                CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                context.dagCache.size));

    ////////////////////////////////////////////////////////////////////////////
    if (   false == parameters.headerCache.alloc(clQueue, *clContext)
        || false == parameters.resultCache.alloc(clQueue, *clContext))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (   nullptr == parameters.lightCache
        || nullptr == parameters.dagCache)
    {
        logErr() << "Fail to alloc memory";
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    OPENCL_ER(clQueue->enqueueWriteBuffer(
        *parameters.lightCache,
        CL_TRUE,
        0,
        context.lightCache.size,
        context.lightCache.hash));

    ////////////////////////////////////////////////////////////////////////////
    if (   false == buildDAG()
        || false == buildSearch())
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverAmdEthash::updateConstants(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    uint32_t const* const header { jobInfo.headerHash.word32 };
    if (false == parameters.headerCache.setBufferDevice(clQueue, header))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    setBlocks(getMaxGroupSize());
    setThreads(8192u);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverAmdEthash::buildDAG()
{
    ////////////////////////////////////////////////////////////////////////////
    // Clear old data
    kernelGenerator.clear();

    ////////////////////////////////////////////////////////////////////////////
    // kernel name
    kernelGenerator.setKernelName("ethash_build_dag");

    ////////////////////////////////////////////////////////////////////////////
    // defines
    kernelGenerator.addDefine("GROUP_SIZE", castU32(clDevice->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()));
    kernelGenerator.addDefine("DAG_LOOP", algo::ethash::DAG_ITEM_PARENTS / 4u / 4u);

    ////////////////////////////////////////////////////////////////////////////
    // ethash files
    if (false == kernelGenerator.appendFile("kernel/ethash/ethash_dag.cl"))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // build opencl kernel
    if (false == kernelGenerator.buildOpenCL(clDevice, clContext))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Set kernel parameters
    auto& clKernel { kernelGenerator.clKernel };
    OPENCL_ER(clKernel.setArg(0u, *(parameters.dagCache)));
    OPENCL_ER(clKernel.setArg(1u, *(parameters.lightCache)));
    OPENCL_ER(clKernel.setArg(2u, algo::ethash::DAG_ITEM_PARENTS));
    OPENCL_ER(clKernel.setArg(3u, castU32(context.dagCache.numberItem)));
    OPENCL_ER(clKernel.setArg(4u, castU32(context.lightCache.numberItem)));

    ////////////////////////////////////////////////////////////////////////////
    // Run kernel to build DAG
    uint32_t const maxGroupSize { getMaxGroupSize() };
    uint32_t const threadKernel { castU32(context.dagCache.numberItem) / maxGroupSize };
    OPENCL_ER(
        clQueue->enqueueNDRangeKernel(
            clKernel,
            cl::NullRange,
            cl::NDRange(maxGroupSize, threadKernel, 1),
            cl::NDRange(maxGroupSize, 1,            1)));
    OPENCL_ER(clQueue->finish());

    return true;
}


bool resolver::ResolverAmdEthash::buildSearch()
{
    ////////////////////////////////////////////////////////////////////////////
    // Clear old data
    kernelGenerator.clear();

    ////////////////////////////////////////////////////////////////////////////
    // kernel name
    kernelGenerator.setKernelName("ethash_search");

    ////////////////////////////////////////////////////////////////////////////
    // defines
    uint32_t const groupSize { getMaxGroupSize() };
    uint32_t const laneParallel { 8u };
    uint32_t const groupParallel { groupSize / laneParallel };
    uint32_t const lenSeed { 4u };
    uint32_t const lenState { 25u };

    kernelGenerator.addDefine("GROUP_SIZE", groupSize);
    kernelGenerator.addDefine("DAG_NUMBER_ITEM", context.dagCache.numberItem);
    kernelGenerator.addDefine("LANE_PARALLEL", laneParallel);
    kernelGenerator.addDefine("LEN_SEED", lenSeed);
    kernelGenerator.addDefine("LEN_STATE", lenState);
    kernelGenerator.addDefine("LEN_HASHES", groupParallel * lenSeed);
    kernelGenerator.addDefine("LEN_WORD0", groupSize);
    kernelGenerator.addDefine("LEN_REDUCE", groupSize);
    kernelGenerator.addDefine("LEN_SWAPPER", groupParallel);
    kernelGenerator.addDefine("LEN_KECCAK", 24u);
    kernelGenerator.addDefine("MAX_KECCAK_ROUND", 23u);

    ////////////////////////////////////////////////////////////////////////////
    // ethash files
    kernelGenerator.appendFile("kernel/ethash/ethash_search.cl");

    ////////////////////////////////////////////////////////////////////////////
    // build opencl kernel
    if (false == kernelGenerator.buildOpenCL(clDevice, clContext))
    {
        return false;
    }

    return true;
}

bool resolver::ResolverAmdEthash::execute(
    stratum::StratumJobInfo const& jobInfo)
{
    auto& clKernel { kernelGenerator.clKernel };
    OPENCL_ER(clKernel.setArg(0u, *(parameters.dagCache)));
    OPENCL_ER(clKernel.setArg(1u, *(parameters.resultCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(2u, *(parameters.headerCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(3u, jobInfo.nonce));
    OPENCL_ER(clKernel.setArg(4u, jobInfo.boundaryU64));

    OPENCL_ER(
        clQueue->enqueueNDRangeKernel(
            clKernel,
            cl::NullRange,
            cl::NDRange(blocks, threads, 1),
            cl::NDRange(blocks, 1,       1)));
    OPENCL_ER(clQueue->finish());

    if (false == getResultCache(jobInfo.jobIDStr, jobInfo.extraNonceSize))
    {
        return false;
    }

    return true;
}


bool resolver::ResolverAmdEthash::getResultCache(
    std::string const& jobId,
    uint32_t const extraNonceSize)
{
    algo::ethash::Result data{};

    if (false == parameters.resultCache.getBufferHost(clQueue, &data))
    {
        return false;
    }

    if (true == data.found)
    {
        resultShare.found = data.found;
        resultShare.count = data.count;
        resultShare.extraNonceSize = extraNonceSize;
        resultShare.jobId.assign(jobId);

        for (uint32_t i { 0u }; i < data.count; ++i)
        {
            resultShare.nonces[i] = data.nonces[i];
        }

        if (false == parameters.resultCache.resetBufferHost(clQueue))
        {
            return false;
        }
    }

    return true;
}


void resolver::ResolverAmdEthash::submit(
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
                    nonceHexa.str().substr(resultShare.extraNonceSize)
                };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }

        resultShare.count = 0u;
        resultShare.found = false;
    }
}
