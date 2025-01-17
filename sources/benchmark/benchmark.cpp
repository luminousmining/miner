#include <algo/autolykos/autolykos.hpp>
#include <algo/hash_utils.hpp>
#include <benchmark/benchmark.hpp>
#include <benchmark/cuda/kernels.hpp>
#include <common/formater_hashrate.hpp>
#include <common/log/log.hpp>
#include <common/custom.hpp>


benchmark::Benchmark::Benchmark(
    bool const nvidia,
    bool const amd)
{
    enableNvidia = nvidia;
    enableAmd = amd;
}


void benchmark::Benchmark::initializeDevices(uint32_t const deviceIndex)
{
    if (true == enableNvidia)
    {
        benchmark::initializeCuda(propertiesNvidia, deviceIndex);
    }
    if (true == enableAmd)
    {
        benchmark::initializeOpenCL(propertiesAmd, deviceIndex);
    }
}


void benchmark::Benchmark::destroyDevices()
{
    if (true == enableNvidia)
    {
        benchmark::cleanUpCuda();
    }
    if (true == enableAmd)
    {
        benchmark::cleanUpOpenCL(propertiesAmd);
    }
}


void benchmark::Benchmark::run()
{
    if (true == enableNvidia)
    {
        runNvidia();
    }
    if (true == enableAmd)
    {
        runAmd();
    }
}


void benchmark::Benchmark::startChrono(std::string const& benchName)
{
    currentBenchName = benchName;
    stats.setChronoUnit(common::CHRONO_UNIT::US);
    stats.setBatchNonce(nonceComputed);
    stats.reset();
}


void benchmark::Benchmark::stopChrono()
{
    stats.increaseKernelExecuted();
    stats.stop();
    stats.updateHashrate();
    double const hashrate{ stats.getHashrate() };
    logInfo() << currentBenchName << ": " << common::hashrateToString(hashrate);
}


bool benchmark::Benchmark::initCleanResult(t_result** result)
{
    CU_ALLOC_HOST(result, sizeof(t_result));

    (*result)->found = false;
    (*result)->count = 0u;
    (*result)->nonce = 0ull;

    return true;
}


bool benchmark::Benchmark::initCleanResult32(t_result_32** result)
{
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

    return true;
}


bool benchmark::Benchmark::initCleanResult64(t_result_64** result)
{
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

    return true;
}


void benchmark::Benchmark::runNvidia()
{
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


void benchmark::Benchmark::runAmd()
{
}
