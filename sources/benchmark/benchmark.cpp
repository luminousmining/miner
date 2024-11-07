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
    runNvidiaEthash();
}


void benchmark::Benchmark::runNvidiaEthash()
{
    ////////////////////////////////////////////////////////////////////////////
    uint32_t* ethashDag{ nullptr };
    uint64_t const dagItems{ 45023203ull };
    CU_ALLOC(&ethashDag, sizeof(uint32_t) * dagItems);
    init_array(ethashDag, dagItems);

    std::string const header{ "257cf0c2c67dd2c39842da75f97dc76d41c7cbaf31f71d5d387b16cbf3da730b" };

    ////////////////////////////////////////////////////////////////////////////
    /* etherminer implementation */
    init_ethash_v0(header);
    ethash_v0(propertiesNvidia.cuStream, 8192u, 256u);

    ////////////////////////////////////////////////////////////////////////////
    CU_SAFE_DELETE(ethashDag);
}


void benchmark::Benchmark::runAmd()
{
}
