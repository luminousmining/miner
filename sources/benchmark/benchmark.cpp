#include <cuda.h>
#include <cuda_runtime.h>

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
    stats.setChronoUnit(common::CHRONO_UNIT::MS);
    stats.setBatchNonce(blocks * threads);
    stats.reset();
    stats.start();
}


void benchmark::Benchmark::stopChrono()
{
    stats.increaseKernelExecuted();
    stats.stop();
    stats.updateHashrate();
    double const hashrate{ stats.getHashrate() };
    logInfo() << currentBenchName << ": " << common::hashrateToString(hashrate);
}


void benchmark::Benchmark::runNvidia()
{
    if (false == runNvidiaEthash())
    {
        logErr() << "Nvidia ETHASH failled!";
    }
}


bool benchmark::Benchmark::runNvidiaEthash()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    uint64_t const dagItems{ 45023203ull };
    uint64_t const boundary{ 10695475200ull };
    auto const headerHash{ algo::toHash<algo::hash256>("257cf0c2c67dd2c39842da75f97dc76d41c7cbaf31f71d5d387b16cbf3da730b") };

    ////////////////////////////////////////////////////////////////////////////
    algo::hash1024* dagHash{ nullptr };
    CU_ALLOC(&dagHash, dagItems * algo::LEN_HASH_1024);
    if (false == init_array(propertiesNvidia.cuStream, dagHash->word32, dagItems))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    t_result_64* result{ nullptr };
    CU_ALLOC_HOST(&result, sizeof(t_result_64));
    result->error = false;
    result->found = false;
    for (uint32_t i{ 0u }; i < MAX_RESULT_INDEX; ++i)
    {
        result->nonce[i] = 0ull;
    }
    for (uint32_t x{ 0u }; x < MAX_RESULT_INDEX; ++x)
    {
        for (uint32_t y{ 0u }; y < MAX_RESULT_INDEX; ++y)
        {
            result->mix[x][y] = 0ull;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    blocks = 8192u;
    threads = 256u;

    ////////////////////////////////////////////////////////////////////////////
    if (true == init_ethash_ethminer(dagHash, &headerHash, dagItems, boundary))
    {
        startChrono("ethash_ethminer"s);
        ethash_ethminer(propertiesNvidia.cuStream, result, blocks, threads);
        stopChrono();
    }

    ////////////////////////////////////////////////////////////////////////////
    CU_SAFE_DELETE(dagHash);

    return true;
}


void benchmark::Benchmark::runAmd()
{
}
