#include <boost/json.hpp>
#include <fstream>

#include <algo/autolykos/autolykos.hpp>
#include <algo/hash_utils.hpp>
#include <benchmark/workflow.hpp>
#include <benchmark/cuda/kernels.hpp>
#include <common/formater_hashrate.hpp>
#include <common/log/log.hpp>
#include <common/custom.hpp>


benchmark::BenchmarkWorkflow::BenchmarkWorkflow(
    [[maybe_unused]] bool const nvidia,
    [[maybe_unused]] bool const amd)
{
#if defined(CUDA_ENABLE)
    enableNvidia = nvidia;
#endif
#if defined(AMD_ENABLE)
    enableAmd = amd;
#endif
}


bool benchmark::BenchmarkWorkflow::initializeDevices(uint32_t const deviceIndex)
{
#if defined(CUDA_ENABLE)
    if (true == enableNvidia)
    {
        if (false == benchmark::initializeCuda(propertiesNvidia, deviceIndex))
        {
            logErr() << "Fail to load device NVIDIA";
            return false;
        }
    }
#endif
#if defined(AMD_ENABLE)
    if (true == enableAmd)
    {
        if (false == benchmark::initializeOpenCL(propertiesAmd, deviceIndex))
        {
            logErr() << "Fail to load device AMD";
            return false;
        }
    }
#endif
    return true;
}


void benchmark::BenchmarkWorkflow::destroyDevices()
{
#if defined(CUDA_ENABLE)
    if (true == enableNvidia)
    {
        benchmark::cleanUpCuda();
    }
#endif
#if defined(AMD_ENABLE)
    if (true == enableAmd)
    {
        benchmark::cleanUpOpenCL(propertiesAmd);
    }
#endif
}


void benchmark::BenchmarkWorkflow::run()
{
#if defined(CUDA_ENABLE)
    if (true == enableNvidia)
    {
        runNvidia();
    }
#endif
#if defined(AMD_ENABLE)
    if (true == enableAmd)
    {
        runAmd();
    }
#endif
    writeReport();
}


void benchmark::BenchmarkWorkflow::setMultiplicator(uint32_t const _multiplicator)
{
    multiplicator = _multiplicator;
}


void benchmark::BenchmarkWorkflow::setDivisor(uint32_t const _divisor)
{
    divisor = _divisor;
}


void benchmark::BenchmarkWorkflow::setGrid(uint32_t const _threads, uint32_t _blocks)
{
    threads = _threads;
    blocks = _blocks;
    nonceComputed = blocks * threads;
}


void benchmark::BenchmarkWorkflow::startChrono(
    std::string const& benchName)
{
    currentBenchName = benchName;
    stats.setChronoUnit(common::CHRONO_UNIT::US);
    stats.setBatchNonce(nonceComputed);
    stats.reset();
}


void benchmark::BenchmarkWorkflow::stopChrono(uint32_t const index)
{
    ////////////////////////////////////////////////////////////////////////////
    stats.increaseKernelExecuted();
    stats.stop();
    stats.updateHashrate();
    double const hashrate{ (stats.getHashrate() * multiplicator) / divisor };
    logInfo() << currentBenchName << ": " << common::hashrateToString(hashrate);

    ////////////////////////////////////////////////////////////////////////////
    if (index == 0u)
    {
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    benchmark::Snapshot snapshot{};
    snapshot.deviceType = currentdeviceType;
    snapshot.name = currentBenchName;
    snapshot.threads = threads;
    snapshot.blocks = blocks;
    snapshot.perform = hashrate;

    ////////////////////////////////////////////////////////////////////////////
    snapshots.emplace_back(snapshot);
}


void benchmark::BenchmarkWorkflow::writeReport()
{
     boost::json::object root{};
#if defined(CUDA_ENABLE)
     boost::json::array nvidia{};
#endif
#if defined(AMD_ENABLE)
     boost::json::array amd{};
#endif

    for (auto const& snapshot : snapshots)
    {
        boost::json::object data{};
        data["name"] = snapshot.name;
        data["threads"] = snapshot.threads;
        data["blocks"] = snapshot.blocks;
        data["perform"] = snapshot.perform;
        switch(snapshot.deviceType)
        {
#if defined(CUDA_ENABLE)
            case device::DEVICE_TYPE::NVIDIA:
            {
                nvidia.push_back(data);
                break;
            }
#endif
#if defined(AMD_ENABLE)
            case device::DEVICE_TYPE::AMD:
            {
                amd.push_back(data);
                break;
            }
#endif
            case device::DEVICE_TYPE::UNKNOWN:
            {
                break;
            }
        }
    }

#if defined(CUDA_ENABLE)
    root["nvidia"] = nvidia;
#endif
#if defined(AMD_ENABLE)
    root["amd"] = amd;
#endif
    
    std::ofstream output{ "benchmark.json" };
    if (output.is_open())
    {
        output << boost::json::serialize(root);
        output.close();
        logInfo() << "Writen report benchmark.json";
    }
}


bool benchmark::BenchmarkWorkflow::initCleanResult(
    [[maybe_unused]] t_result** result)
{
#if defined(CUDA_ENABLE)
    if (nullptr == *result)
    {
        CU_ALLOC_HOST(result, sizeof(t_result));
    }

    (*result)->found = false;
    (*result)->count = 0u;
    (*result)->nonce = 0ull;
#endif

    return true;
}


bool benchmark::BenchmarkWorkflow::initCleanResult32(
    [[maybe_unused]] t_result_32** result)
{
#if defined(CUDA_ENABLE)
    if (nullptr == *result)
    {
        CU_ALLOC_HOST(result, sizeof(t_result_32));
    }

    (*result)->error = false;
    (*result)->found = false;

    for (uint32_t i{ 0u }; i < MAX_RESULT_INDEX; ++i)
    {
        (*result)->nonce[i] = 0ull;
    }

    for (uint32_t x{ 0u }; x < MAX_RESULT_INDEX; ++x)
    {
        for (uint32_t y{ 0u }; y < MAX_RESULT_INDEX; ++y)
        {
            (*result)->mix[x][y] = 0ull;
        }
    }
#endif

    return true;
}


bool benchmark::BenchmarkWorkflow::initCleanResult64(
    [[maybe_unused]] t_result_64** result)
{
#if defined(CUDA_ENABLE)
    if (nullptr == *result)
    {
        CU_ALLOC_HOST(result, sizeof(t_result_64));
    }

    (*result)->error = false;
    (*result)->found = false;

    for (uint32_t i{ 0u }; i < MAX_RESULT_INDEX; ++i)
    {
        (*result)->nonce[i] = 0ull;
    }

    for (uint32_t x{ 0u }; x < MAX_RESULT_INDEX; ++x)
    {
        for (uint32_t y{ 0u }; y < MAX_RESULT_INDEX; ++y)
        {
            (*result)->mix[x][y] = 0ull;
        }
    }
#endif

    return true;
}


#if defined(CUDA_ENABLE)
void benchmark::BenchmarkWorkflow::runNvidia()
{
    ///////////////////////////////////////////////////////////////////////////
    logInfo() << "Run benchmark on NVIDIA";

    ///////////////////////////////////////////////////////////////////////////
    currentdeviceType = device::DEVICE_TYPE::NVIDIA;

    ///////////////////////////////////////////////////////////////////////////
    if (false == runNvidiaKeccak())
    {
        logErr() << "Nvidia Keccak failled!";
    }
    if (false == runNvidiaFnv1())
    {
        logErr() << "Nvidia Keccak failled!";
    }
    if (false == runNvidiaEthashLightCache())
    {
        logErr() << "Nvidia ETHASH failled!";
    }
    if (false == runNvidiaEthash())
    {
        logErr() << "Nvidia ETHASH failled!";
    }
    if (false == runNvidiaAutolykosv2())
    {
        logErr() << "Nvidia AutolykosV2 failled!";
    }
    if (false == runNvidiaKawpow())
    {
        logErr() << "Nvidia Kawpow failled!";
    }
}
#endif


#if defined(AMD_ENABLE)
void benchmark::BenchmarkWorkflow::runAmd()
{
    ///////////////////////////////////////////////////////////////////////////
    logInfo() << "Run benchmark on AMD";

    ///////////////////////////////////////////////////////////////////////////
    currentdeviceType = device::DEVICE_TYPE::AMD;
}
#endif
