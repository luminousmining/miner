#include <boost/json.hpp>
#include <fstream>

#include <algo/autolykos/autolykos.hpp>
#include <algo/hash_utils.hpp>
#include <benchmark/benchmark.hpp>
#include <benchmark/cuda/kernels.hpp>
#include <common/formater_hashrate.hpp>
#include <common/log/log.hpp>
#include <common/custom.hpp>


benchmark::Benchmark::Benchmark(
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


void benchmark::Benchmark::initializeDevices(uint32_t const deviceIndex)
{
#if defined(CUDA_ENABLE)
    if (true == enableNvidia)
    {
        benchmark::initializeCuda(propertiesNvidia, deviceIndex);
    }
#endif
#if defined(AMD_ENABLE)
    if (true == enableAmd)
    {
        benchmark::initializeOpenCL(propertiesAmd, deviceIndex);
    }
#endif
}


void benchmark::Benchmark::destroyDevices()
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


void benchmark::Benchmark::run()
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


void benchmark::Benchmark::setMultiplicator(uint32_t const _multiplicator)
{
    multiplicator = _multiplicator;
}


void benchmark::Benchmark::setGrid(uint32_t const _threads, uint32_t _blocks)
{
    threads = _threads;
    blocks = _blocks;
    nonceComputed = blocks * threads;
}


void benchmark::Benchmark::startChrono(
    uint32_t const index,
    std::string const& benchName)
{
    if (index == 0u)
    {
        return;
    }
    currentBenchName = benchName;
    stats.setChronoUnit(common::CHRONO_UNIT::US);
    stats.setBatchNonce(nonceComputed);
    stats.reset();
}


void benchmark::Benchmark::stopChrono(uint32_t const index)
{
    ////////////////////////////////////////////////////////////////////////////
    if (index == 0u)
    {
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    stats.increaseKernelExecuted();
    stats.stop();
    stats.updateHashrate();
    double const hashrate{ stats.getHashrate() * multiplicator };
    logInfo() << currentBenchName << ": " << common::hashrateToString(hashrate);

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


void benchmark::Benchmark::writeReport()
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
            case device::DEVICE_TYPE::UNKNOW:
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


bool benchmark::Benchmark::initCleanResult(
    [[maybe_unused]] t_result** result)
{
#if defined(CUDA_ENABLE)
    CU_ALLOC_HOST(result, sizeof(t_result));

    (*result)->found = false;
    (*result)->count = 0u;
    (*result)->nonce = 0ull;
#endif

    return true;
}


bool benchmark::Benchmark::initCleanResult32(
    [[maybe_unused]] t_result_32** result)
{
#if defined(CUDA_ENABLE)
    CU_ALLOC_HOST(result, sizeof(t_result_32));

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


bool benchmark::Benchmark::initCleanResult64(
    [[maybe_unused]] t_result_64** result)
{
#if defined(CUDA_ENABLE)
    CU_ALLOC_HOST(result, sizeof(t_result_64));

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
void benchmark::Benchmark::runNvidia()
{
    currentdeviceType = device::DEVICE_TYPE::NVIDIA;
    // if (false == runNvidiaEthash())
    // {
    //     logErr() << "Nvidia ETHASH failled!";
    // }
    // if (false == runNvidiaAutolykosv2())
    // {
    //     logErr() << "Nvidia AutolykosV2 failled!";
    // }
    if (false == runNvidiaKawpow())
    {
        logErr() << "Nvidia Kawpow failled!";
    }
}
#endif

#if defined(AMD_ENABLE)
void benchmark::Benchmark::runAmd()
{
    currentdeviceType = device::DEVICE_TYPE::AMD;
}
#endif
