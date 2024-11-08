#include <cuda.h>
#include <cuda_runtime.h>

#include <benchmark/benchmark.hpp>
#include <benchmark/cuda/kernels.hpp>
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
    /* etherminer implementation */
    if (true == init_ethash_v0(dagHash, &headerHash, dagItems, boundary))
    {
        ethash_v0(propertiesNvidia.cuStream, 1u, 8u);
    }

    ////////////////////////////////////////////////////////////////////////////
    CU_SAFE_DELETE(dagHash);

    return true;
}


void benchmark::Benchmark::runAmd()
{
}
