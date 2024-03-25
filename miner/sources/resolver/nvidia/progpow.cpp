#include <algo/hash_utils.hpp>
#include <algo/ethash/ethash.hpp>
#include <common/cast.hpp>
#include <common/error/cuda_error.hpp>
#include <common/log/log.hpp>
#include <resolver/nvidia/progpow.hpp>

#include <algo/progpow/cuda/progpow.cuh>


resolver::ResolverNvidiaProgPOW::~ResolverNvidiaProgPOW()
{
    progpowFreeMemory(parameters);
}


bool resolver::ResolverNvidiaProgPOW::updateContext(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    algo::ethash::initializeDagContext(context,
                                       jobInfo.epoch,
                                       maxEpoch,
                                       dagCountItemsGrowth,
                                       dagCountItemsInit);

    if (   context.lightCache.numberItem == 0ull
        || context.lightCache.size == 0ull
        || context.dagCache.numberItem == 0ull
        || context.dagCache.size == 0ull)
    {
        logErr()
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

    return true;
}


bool resolver::ResolverNvidiaProgPOW::updateMemory(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    if (false == updateContext(jobInfo))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == progpowInitMemory(context, parameters))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == progpowBuildDag(getCurrentStream(),
                                 dagItemParents,
                                 castU32(context.dagCache.numberItem)))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool resolver::ResolverNvidiaProgPOW::buildSearch()
{
    ////////////////////////////////////////////////////////////////////////////
    algo::progpow::writeMathRandomKernelCuda(progpowVersion,
                                             deviceId,
                                             currentPeriod,
                                             countCache,
                                             countMath);

    ////////////////////////////////////////////////////////////////////////////
    switch (progpowVersion)
    {
        case algo::progpow::VERSION::V_0_9_2:
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
    uint32_t const dagSize { castU32(context.dagCache.numberItem / 2ull) };
    kernelGenerator.addDefine("DAG_SIZE", dagSize);
    kernelGenerator.addDefine("COUNT_DAG", algo::progpow::COUNT_DAG);
    kernelGenerator.addDefine("STATE_LEN", 25u);

    ////////////////////////////////////////////////////////////////////////////
    kernelGenerator.appendLine("using uint32_t = unsigned int;");
    kernelGenerator.appendLine("using uint64_t = unsigned long long;");

    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;
    std::string const fileSequenceMathPeriod
    {
        "kernel/progpow/sequence_math_random"s
        + "_"s + std::to_string(deviceId)
        + "_"s + std::to_string(currentPeriod)
        + ".cuh"s
    };
    std::string kernelDerived{};
    switch (progpowVersion)
    {
        case algo::progpow::VERSION::V_0_9_2: /* algo::progpow::VERSION::V_0_9_3 */
        case algo::progpow::VERSION::V_0_9_3: kernelDerived.assign("progpow_functions.cuh"); break;
        case algo::progpow::VERSION::KAWPOW: kernelDerived.assign("kawpow_functions.cuh"); break;
        case algo::progpow::VERSION::FIROPOW: kernelDerived.assign("firopow_functions.cuh"); break;
        case algo::progpow::VERSION::EVRPROGPOW: kernelDerived.assign("evrprogpow_functions.cuh"); break;
    }
    if (   false == kernelGenerator.appendFile("kernel/common/be_u32.cuh")
        || false == kernelGenerator.appendFile("kernel/common/be_u64.cuh")
        || false == kernelGenerator.appendFile("kernel/common/copy_u4.cuh")
        || false == kernelGenerator.appendFile("kernel/common/get_lane_id.cuh")
        || false == kernelGenerator.appendFile("kernel/common/register.cuh")
        || false == kernelGenerator.appendFile("kernel/common/rotate_byte.cuh")
        || false == kernelGenerator.appendFile("kernel/common/to_u4.cuh")
        || false == kernelGenerator.appendFile("kernel/common/to_u64.cuh")
        || false == kernelGenerator.appendFile("kernel/common/xor.cuh")
        || false == kernelGenerator.appendFile("kernel/crypto/fnv1.cuh")
        || false == kernelGenerator.appendFile("kernel/crypto/keccak_f800.cuh")
        || false == kernelGenerator.appendFile("kernel/crypto/kiss99.cuh")
        || false == kernelGenerator.appendFile("kernel/progpow/" + kernelDerived)
        || false == kernelGenerator.appendFile(fileSequenceMathPeriod)
        || false == kernelGenerator.appendFile("kernel/progpow/result.hpp")
        || false == kernelGenerator.appendFile("kernel/progpow/search.cu"))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    kernelGenerator.setKernelName("progpowSearch");

    IS_NULL(cuProperties);

    ////////////////////////////////////////////////////////////////////////////
    if (false == kernelGenerator.buildCuda(castU32(cuProperties->major),
                                           castU32(cuProperties->minor)))
    {
        return false;
    }

    return true;
}


bool resolver::ResolverNvidiaProgPOW::updateConstants(
    stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    if (currentPeriod != jobInfo.period)
    {
        currentPeriod = jobInfo.period;
        logInfo() << "Build period " << currentPeriod;

        if (false == buildSearch())
        {
            return false;
        }

        setThreads(256u);
        setBlocks(4096u);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == progpowUpdateConstants(jobInfo.headerHash.word32,
                                        parameters.headerCache))
    {
        return false;
    }

    return true;
}


bool resolver::ResolverNvidiaProgPOW::execute(
    stratum::StratumJobInfo const& jobInfo)
{
    IS_NULL(parameters.lightCache);
    IS_NULL(parameters.dagCache);

    if (false == isDoubleStream)
    {
        ////////////////////////////////////////////////////////////////////////
        algo::progpow::Result& paramResultCache { parameters.resultCache[getCurrentIndex()] };

        ////////////////////////////////////////////////////////////////////////
        uint64_t nonce { jobInfo.nonce };
        uint64_t boundary { jobInfo.boundaryU64 };
        void* arguments[]
        {
            &nonce,
            &boundary,
            &parameters.headerCache,
            &parameters.dagCache,
            &paramResultCache
        };

        ////////////////////////////////////////////////////////////////////////////
        CU_ER(cuLaunchKernel(kernelGenerator.cuFunction,
                            blocks,  1u, 1u,
                            threads, 1u, 1u,
                            0u,
                            getCurrentStream(),
                            arguments,
                            nullptr));
        CUDA_ER(cudaStreamSynchronize(getCurrentStream()));
        CUDA_ER(cudaGetLastError());

        ////////////////////////////////////////////////////////////////////////////
        if (true == paramResultCache.found)
        {
            uint32_t const count { MAX_LIMIT(paramResultCache.count, 4u) };

            resultShare.found = true;
            resultShare.count = count;
            resultShare.jobId = jobInfo.jobIDStr;

            for (uint32_t i { 0u }; i < count; ++i)
            {
                resultShare.nonces[i] = paramResultCache.nonces[i];
                for (uint32_t j { 0u }; j < algo::LEN_HASH_256_WORD_32; ++j)
                {
                    resultShare.hash[i][j] = paramResultCache.hash[i][j];
                }
            }

            paramResultCache.found = false;
            paramResultCache.count = 0u;
        }
    }
    else
    {
        ////////////////////////////////////////////////////////////////////////
        CUDA_ER(cudaStreamSynchronize(getCurrentStream()));
        CUDA_ER(cudaGetLastError());

        ////////////////////////////////////////////////////////////////////////
        algo::progpow::Result& paramResultCacheNext { parameters.resultCache[getNextIndex()] };
        uint64_t nonce { jobInfo.nonce };
        uint64_t boundary { jobInfo.boundaryU64 };
        void* arguments[]
        {
            &nonce,
            &boundary,
            &parameters.headerCache,
            &parameters.dagCache,
            &paramResultCacheNext
        };
        CU_ER(cuLaunchKernel(kernelGenerator.cuFunction,
                             blocks,  1u, 1u,
                             threads, 1u, 1u,
                             0u,
                             getNextStream(),
                             arguments,
                             nullptr));

        ////////////////////////////////////////////////////////////////////////
        algo::progpow::Result& paramResultCache { parameters.resultCache[getCurrentIndex()] };
        if (true == paramResultCache.found)
        {
            uint32_t const count { MAX_LIMIT(paramResultCache.count, 4u) };

            resultShare.found = true;
            resultShare.count = count;
            resultShare.jobId = jobInfo.jobIDStr;

            for (uint32_t i { 0u }; i < count; ++i)
            {
                resultShare.nonces[i] = paramResultCache.nonces[i];
                for (uint32_t j { 0u }; j < algo::LEN_HASH_256_WORD_32; ++j)
                {
                    resultShare.hash[i][j] = paramResultCache.hash[i][j];
                }
            }

            paramResultCache.found = false;
            paramResultCache.count = 0u;
        }

        ////////////////////////////////////////////////////////////////////////
        swapStream();
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


void resolver::ResolverNvidiaProgPOW::submit(
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


void resolver::ResolverNvidiaProgPOW::submit(
    stratum::StratumSmartMining* const stratum)
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
