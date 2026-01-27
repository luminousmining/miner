#include <CL/opencl.hpp>

#include <algo/keccak.hpp>
#include <algo/ethash/ethash.hpp>
#include <common/cast.hpp>
#include <common/chrono.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <common/error/opencl_error.hpp>
#include <resolver/amd/ethash.hpp>


resolver::ResolverAmdEthash::~ResolverAmdEthash()
{
    parameters.lightCache.free();
    parameters.dagCache.free();
    parameters.headerCache.free();
    parameters.resultCache.free();

}


bool resolver::ResolverAmdEthash::updateContext(
    stratum::StratumJobInfo const& jobInfo)
{
    algo::ethash::initializeDagContext(context,
                                       jobInfo.epoch,
                                       algo::ethash::MAX_EPOCH_NUMBER,
                                       dagCountItemsGrowth,
                                       dagCountItemsInit,
                                       lightCacheCountItemsGrowth,
                                       lightCacheCountItemsInit);

    if (   context.lightCache.numberItem == 0ull
        || context.lightCache.size == 0ull
        || context.dagCache.numberItem == 0ull
        || context.dagCache.size == 0ull)
    {
        resolverErr()
            << "\n"
            << "=========================================================================" << "\n"
            << "context.lightCache.numberItem: " << context.lightCache.numberItem << "\n"
            << "context.lightCache.size: " << context.lightCache.size << "\n"
            << "context.dagCache.numberItem: " << context.dagCache.numberItem << "\n"
            << "context.dagCache.size: " << context.dagCache.size << "\n"
            << "=========================================================================" << "\n"
            ;
        return false;
    }

    uint64_t const totalMemoryNeeded{ (context.dagCache.size + context.lightCache.size) };
    if (   0ull != deviceMemoryAvailable
        && totalMemoryNeeded >= deviceMemoryAvailable)
    {
        resolverErr()
            << "Device have not memory size available."
            << " Needed " << totalMemoryNeeded << ", memory available " << deviceMemoryAvailable;
        return false;
    }

    return true;
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
    if (false == updateContext(jobInfo))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    parameters.lightCache.free();
    parameters.dagCache.free();
    parameters.headerCache.free();
    parameters.resultCache.free();

    ////////////////////////////////////////////////////////////////////////////
    parameters.lightCache.setSize(context.lightCache.size);
    parameters.dagCache.setSize(context.dagCache.size);

    ////////////////////////////////////////////////////////////////////////////
    if (   false == parameters.lightCache.alloc(*clContext)
        || false == parameters.dagCache.alloc(*clContext)
        || false == parameters.headerCache.alloc(clQueue, *clContext)
        || false == parameters.resultCache.alloc(clQueue, *clContext))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == parameters.lightCache.write(context.lightCache.hash,
                                             context.lightCache.size,
                                             clQueue))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (   false == buildDAG()
        || false == buildSearch())
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    algo::ethash::freeDagContext(context);

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
    overrideOccupancy(8192u, getMaxGroupSize());


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
    if (false == kernelGenerator.build(clDevice, clContext))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Set kernel parameters
    auto& clKernel { kernelGenerator.clKernel };
    OPENCL_ER(clKernel.setArg(0u, *(parameters.dagCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(1u, *(parameters.lightCache.getBuffer())));
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

    ////////////////////////////////////////////////////////////////////////////
    parameters.lightCache.free();

    ////////////////////////////////////////////////////////////////////////////
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
    if (false == kernelGenerator.build(clDevice, clContext))
    {
        return false;
    }

    return true;
}

bool resolver::ResolverAmdEthash::executeSync(
    stratum::StratumJobInfo const& jobInfo)
{
    auto& clKernel { kernelGenerator.clKernel };
    OPENCL_ER(clKernel.setArg(0u, *(parameters.dagCache.getBuffer())));
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


bool resolver::ResolverAmdEthash::executeAsync(
    stratum::StratumJobInfo const& jobInfo)
{
    return executeSync(jobInfo);
}


bool resolver::ResolverAmdEthash::getResultCache(
    std::string const& _jobId,
    uint32_t const extraNonceSize)
{
    algo::ethash::Result data{};

    if (false == parameters.resultCache.getBufferHost(clQueue, &data))
    {
        return false;
    }

    if (true == data.found)
    {
        uint32_t const count
        {
            common::max_limit(data.count, algo::ethash::MAX_RESULT)
        };

        resultShare.found = true;
        resultShare.count = count;
        resultShare.extraNonceSize = extraNonceSize;
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


void resolver::ResolverAmdEthash::submit(
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
