#include <algo/hash_utils.hpp>
#include <algo/ethash/ethash.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <common/error/opencl_error.hpp>
#include <resolver/amd/progpow.hpp>


void resolver::ResolverAmdProgPOW::updateContext(
    stratum::StratumJobInfo const& jobInfo)
{
    algo::ethash::initializeDagContext(context,
                                       jobInfo.epoch,
                                       maxEpoch,
                                       dagCountItemsGrowth,
                                       dagCountItemsInit);
}


bool resolver::ResolverAmdProgPOW::updateMemory(
    stratum::StratumJobInfo const& jobInfo)
{
    IS_NULL(clContext);
    IS_NULL(clQueue);

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
    IS_NULL(parameters.lightCache);
    IS_NULL(parameters.dagCache);

    ////////////////////////////////////////////////////////////////////////////
    OPENCL_ER(
        clQueue->enqueueWriteBuffer(
            *parameters.lightCache,
            CL_TRUE,
            0,
            context.lightCache.size,
            context.lightCache.hash));

    ////////////////////////////////////////////////////////////////////////////
    if (false == buildDAG())
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverAmdProgPOW::updateConstants(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    if (currentPeriod != jobInfo.period)
    {
        currentPeriod = jobInfo.period;
        logInfo() << "Build period " << currentPeriod;

        ////////////////////////////////////////////////////////////////////////////
        if (false == buildSearch())
        {
            return false;
        }

        ////////////////////////////////////////////////////////////////////////////
        setBlocks(getMaxGroupSize());
        setThreads(4096u);
    }

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const* const header { jobInfo.headerHash.word32 };
    if (false == parameters.headerCache.setBufferDevice(clQueue, header))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverAmdProgPOW::buildDAG()
{
    ////////////////////////////////////////////////////////////////////////////
    // Clear old data
    kernelGenerator.clear();

    ////////////////////////////////////////////////////////////////////////////
    // kernel name
    kernelGenerator.setKernelName("ethash_build_dag");

    ////////////////////////////////////////////////////////////////////////////
    // defines
    kernelGenerator.addDefine("GROUP_SIZE", getMaxGroupSize());
    kernelGenerator.addDefine("DAG_LOOP", dagItemParents / 4u / 4u);

    ////////////////////////////////////////////////////////////////////////////
    // progpow files
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
    OPENCL_ER(clKernel.setArg(2u, dagItemParents));
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
    return true;
}


bool resolver::ResolverAmdProgPOW::buildSearch()
{
    ////////////////////////////////////////////////////////////////////////////
    algo::progpow::writeMathRandomKernelOpenCL(deviceId,
                                               currentPeriod,
                                               countCache,
                                               countMath);

    ////////////////////////////////////////////////////////////////////////////
    kernelGenerator.clear();

    ////////////////////////////////////////////////////////////////////////////
    kernelGenerator.setKernelName("progpow_search");

    ////////////////////////////////////////////////////////////////////////////
    switch (progpowVersion)
    {
        case algo::progpow::VERSION::V_0_9_2:
        {
            kernelGenerator.declareDefine("__KERNEL_PROGPOW");
            break;
        }
        case algo::progpow::VERSION::V_0_9_3:
        {
            kernelGenerator.declareDefine("__KERNEL_PROGPOW");
            break;
        }
        case algo::progpow::VERSION::KAWPOW:
        {
            kernelGenerator.declareDefine("__KERNEL_KAWPOW");
            break;
        }
        case algo::progpow::VERSION::FIROPOW:
        {
            kernelGenerator.declareDefine("__KERNEL_FIROPOW");
            break;
        }
        case algo::progpow::VERSION::EVRPROGPOW:
        {
            kernelGenerator.declareDefine("__KERNEL_EVRPROGPOW");
            break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const maxThreadByGroup { getMaxGroupSize() };
    uint32_t const batchGroupLane { getMaxGroupSize() / algo::progpow::LANES };
    kernelGenerator.addDefine("GROUP_SIZE", maxThreadByGroup);
    kernelGenerator.addDefine("MAX_RESULT", 4u);
    kernelGenerator.addDefine("REGS", algo::progpow::REGS);
    kernelGenerator.addDefine("LANES", algo::progpow::LANES);
    kernelGenerator.addDefine("MODULE_CACHE", algo::progpow::MODULE_CACHE);
    kernelGenerator.addDefine("COUNT_DAG", algo::progpow::COUNT_DAG);
    kernelGenerator.addDefine("DAG_SIZE", castU32(context.dagCache.numberItem / 2ull));
    kernelGenerator.addDefine("BATCH_GROUP_LANE", batchGroupLane);
    kernelGenerator.addDefine("SHARE_MSB_LSB_SIZE", 2 * batchGroupLane);
    kernelGenerator.addDefine("SHARE_HASH0_SIZE", batchGroupLane);
    kernelGenerator.addDefine("SHARE_FNV1A_SIZE", maxThreadByGroup);
    kernelGenerator.addDefine("MODULE_CACHE_GROUP", maxThreadByGroup * 4u);
    kernelGenerator.addDefine("MODULE_LOOP", algo::progpow::MODULE_CACHE / (maxThreadByGroup / 4u));

    ////////////////////////////////////////////////////////////////////////////
    kernelGenerator.addInclude("kernel/common/rotate_byte.cl");
    kernelGenerator.addInclude("kernel/crypto/fnv1.cl");
    kernelGenerator.addInclude("kernel/crypto/keccak_f800.cl");
    kernelGenerator.addInclude("kernel/crypto/kiss99.cl");

    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;
    std::string const fileSequenceMathPeriod
    {
        "kernel/progpow/sequence_math_random"s
        + "_"s + std::to_string(deviceId)
        + "_"s + std::to_string(currentPeriod)
        + ".cl"s
    };
    std::string kernelDerived{};
    switch (progpowVersion)
    {
        case algo::progpow::VERSION::V_0_9_2: kernelDerived.assign("progpow_functions.cl"); break;
        case algo::progpow::VERSION::V_0_9_3: kernelDerived.assign("progpow_functions.cl"); break;
        case algo::progpow::VERSION::KAWPOW: kernelDerived.assign("kawpow_functions.cl"); break;
        case algo::progpow::VERSION::FIROPOW: kernelDerived.assign("firopow_functions.cl"); break;
        case algo::progpow::VERSION::EVRPROGPOW: kernelDerived.assign("evrprogpow_functions.cl"); break;
    }
    if (   false == kernelGenerator.appendFile("kernel/progpow/progpow_result.cl")
        || false == kernelGenerator.appendFile("kernel/progpow/" + kernelDerived)
        || false == kernelGenerator.appendFile(fileSequenceMathPeriod)
        || false == kernelGenerator.appendFile("kernel/progpow/progpow.cl"))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == kernelGenerator.buildOpenCL(clDevice, clContext))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}

bool resolver::ResolverAmdProgPOW::execute(
    stratum::StratumJobInfo const& jobInfo)
{
    auto& clKernel { kernelGenerator.clKernel };

    OPENCL_ER(clKernel.setArg(0u, jobInfo.nonce));
    OPENCL_ER(clKernel.setArg(1u, jobInfo.boundaryU64));
    OPENCL_ER(clKernel.setArg(2u, *(parameters.headerCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(3u, *(parameters.dagCache)));
    OPENCL_ER(clKernel.setArg(4u, *(parameters.resultCache.getBuffer())));

    OPENCL_ER(
        clQueue->enqueueNDRangeKernel(
            clKernel,
            cl::NullRange,
            cl::NDRange(blocks, threads, 1),
            cl::NDRange(blocks, 1,       1)));
    OPENCL_ER(clQueue->finish());

    if (false == getResultCache(jobInfo.jobIDStr))
    {
        return false;
    }

    return true;
}


bool resolver::ResolverAmdProgPOW::getResultCache(
    std::string const& jobId)
{
    algo::progpow::Result data{};

    if (false == parameters.resultCache.getBufferHost(clQueue, &data))
    {
        return false;
    }

    if (true == data.found)
    {
        resultShare.found = data.found;
        resultShare.count = data.count;
        resultShare.jobId.assign(jobId);

        for (uint32_t i { 0u }; i < data.count; ++i)
        {
            resultShare.nonces[i] = data.nonces[i];
        }
        for (uint32_t i { 0u }; i < data.count; ++i)
        {
            for (uint32_t j { 0u }; j < algo::LEN_HASH_256_WORD_32; ++j)
            {
                resultShare.hash[i][j] = data.hash[i][j];
            }
        }

        if (false == parameters.resultCache.resetBufferHost(clQueue))
        {
            return false;
        }
    }

    return true;
}


void resolver::ResolverAmdProgPOW::submit(
    stratum::Stratum* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i { 0u }; i < resultShare.count; ++i)
            {
                std::stringstream nonceHexa;
                nonceHexa << "0x" << std::hex << std::setfill('0') << std::setw(16) << resultShare.nonces[i];

                uint32_t hash[algo::LEN_HASH_256_WORD_32]{};
                for (uint32_t j { 0u }; j < algo::LEN_HASH_256_WORD_32; ++j)
                {
                    hash[j] = resultShare.hash[i][j];
                }

                boost::json::array params
                {
                    resultShare.jobId,
                    nonceHexa.str(),
                    "0x" + algo::toHex(algo::toHash256((uint8_t*)hash))
                };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }

        resultShare.count = 0u;
        resultShare.found = false;
    }
}
