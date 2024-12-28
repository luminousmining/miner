#include <cuda.h>
#include <cuda_runtime.h>

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
    stats.setChronoUnit(common::CHRONO_UNIT::MS);
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


bool benchmark::Benchmark::getCleanResult64(t_result_64** result)
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
    if (false == runNvidiaEthash())
    {
        logErr() << "Nvidia ETHASH failled!";
    }
    if (false == runNvidiaAutolykosv2())
    {
        logErr() << "Nvidia AutolykosV2 failled!";
    }
}


bool benchmark::Benchmark::runNvidiaEthash()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    uint64_t const dagItems{ 45023203ull };
    uint64_t const boundary{ 10695475200ull };
    auto const headerHash{ algo::toHash256("257cf0c2c67dd2c39842da75f97dc76d41c7cbaf31f71d5d387b16cbf3da730b") };

    ////////////////////////////////////////////////////////////////////////////
    algo::hash1024* dagHash{ nullptr };
    CU_ALLOC(&dagHash, dagItems * algo::LEN_HASH_1024);
    if (false == init_array(propertiesNvidia.cuStream, dagHash->word32, dagItems))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    t_result_64* result{ nullptr };
    if (false == getCleanResult64(&result))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    blocks = 8192u;
    threads = 256u;
    nonceComputed = blocks * threads;

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


bool benchmark::Benchmark::runNvidiaAutolykosv2()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    t_result_64* result{ nullptr };
    if (false == getCleanResult64(&result))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const height{ 3130463488u };
    uint32_t const period{ 146488965u };
    uint32_t const dagItemCount{ period * algo::autolykos_v2::NUM_SIZE_8 };


    algo::hash256 const header{ algo::toHash256("6f109ba5226d1e0814cdeec79f1231d1d48196b5979a6d816e3621a1ef47ad80") };

    algo::hash256 const boundary
    {
        algo::toHash2<algo::hash256, algo::hash512>(
            algo::toLittleEndian<algo::hash512>(
                algo::decimalToHash<algo::hash512>(
                    "28948022309329048855892746252171976963209391069768726095651290785380")))
    };

    ////////////////////////////////////////////////////////////////////////////
    algo::hash256* headerHash{ nullptr };
    algo::hash256* dagHash{ nullptr };
    algo::hash256* BHashes{ nullptr };

    CU_ALLOC(&headerHash, algo::LEN_HASH_256);
    CU_ALLOC(&dagHash, dagItemCount * algo::LEN_HASH_256);
    CU_ALLOC(&BHashes, algo::autolykos_v2::NONCES_PER_ITER * algo::LEN_HASH_256);

    IS_NULL(headerHash);
    IS_NULL(dagHash);
    IS_NULL(BHashes);

    CUDA_ER(cudaMemcpy((void*)headerHash->bytes,
                       (void const*)header.bytes,
                       algo::LEN_HASH_256,
                       cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////
    threads = 64u;
    blocks = algo::autolykos_v2::NONCES_PER_ITER / 64u;
    nonceComputed = algo::autolykos_v2::NONCES_PER_ITER;

    ////////////////////////////////////////////////////////////////////////////
    if (true == autolykos_v2_mhssamadi_init(boundary))
    {
        if (true == autolykos_v2_mhssamadi_prehash(propertiesNvidia.cuStream,
                                                   dagHash->word32,
                                                   blocks,
                                                   threads,
                                                   period,
                                                   height))
        {
            startChrono("autolykos_v2: mhssamadi"s);
            autolykos_v2_mhssamadi(
                propertiesNvidia.cuStream,
                result,
                dagHash->word32,
                BHashes->word32,
                headerHash->word32,
                blocks,
                threads,
                period,
                height);
            stopChrono();
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == autolykos_v2_init_lm1(boundary))
    {
        if (true == autolykos_v2_prehash_lm1(propertiesNvidia.cuStream,
                                             dagHash->word32,
                                             blocks,
                                             threads,
                                             period,
                                             height))
        {
            startChrono("autolykos_v2: lm1"s);
            autolykos_v2_lm1(propertiesNvidia.cuStream,
                             result,
                             dagHash->word32,
                             headerHash->word32,
                             BHashes->word32,
                             blocks,
                             threads,
                             period);
            stopChrono();
        }
    }

    CU_SAFE_DELETE(headerHash);
    CU_SAFE_DELETE(dagHash);
    CU_SAFE_DELETE(BHashes);

    return true;
}


void benchmark::Benchmark::runAmd()
{
}
