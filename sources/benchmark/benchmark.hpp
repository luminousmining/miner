#pragma once

#include <vector>

#include <benchmark/amd.hpp>
#include <benchmark/nvidia.hpp>
#include <benchmark/result.hpp>
#include <device/type.hpp>
#include <statistical/statistical.hpp>


#define RUN_BENCH(name, loopCount, _threads, _blocks, function)                \
    logInfo() << "================================";                           \
    setGrid(_threads, _blocks);                                                \
    for (uint32_t i{ 0u }; i < loopCount; ++i)                                 \
    {                                                                          \
        startChrono(name);                                                     \
        if (false == (function))                                               \
        {                                                                      \
            return false;                                                      \
        }                                                                      \
        stopChrono(i);                                                         \
    }                                                                          \
    setMultiplicator(1u);                                                      \
    setDivisor(1u);                                                            \
    logInfo() << "================================";


#define BENCH_INIT_RESET_RESULT(result)                                        \
    if (false == initCleanResult(&result))                                     \
    {                                                                          \
        return false;                                                          \
    }

#define BENCH_INIT_RESET_RESULT_32(result)                                     \
    if (false == initCleanResult32(&result))                                   \
    {                                                                          \
        return false;                                                          \
    }

#define BENCH_INIT_RESET_RESULT_64(result)                                     \
    if (false == initCleanResult64(&result))                                   \
    {                                                                          \
        return false;                                                          \
    }

namespace benchmark
{
    struct Snapshot
    {
        device::DEVICE_TYPE deviceType{ device::DEVICE_TYPE::UNKNOW };
        std::string         name{};
        uint32_t            threads{ 0u };
        uint32_t            blocks{ 0u };
        double              perform{ 0.0 };
    };

    struct Benchmark
    {
    public:
#if defined(CUDA_ENABLE)
        bool                        enableNvidia{ true };
        benchmark::PropertiesNvidia propertiesNvidia{};
#endif
#if defined(AMD_ENABLE)
        bool                        enableAmd{ true };
        benchmark::PropertiesAmd    propertiesAmd{};
#endif

        explicit Benchmark(bool const nvidia,
                           bool const amd);
        void initializeDevices(uint32_t const deviceIndex = 0u);
        void destroyDevices();
        void run();

    private:
        device::DEVICE_TYPE              currentdeviceType{ device::DEVICE_TYPE::UNKNOW };
        std::string                      currentBenchName{};
        uint32_t                         blocks{ 1u };
        uint32_t                         threads{ 32u };
        uint64_t                         nonceComputed{ 1ull };
        uint32_t                         multiplicator{ 1u };
        uint32_t                         divisor{ 1u };
        statistical::Statistical         stats{};
        std::vector<benchmark::Snapshot> snapshots{};

        void writeReport();

        void setMultiplicator(uint32_t const _multiplicator);
        void setDivisor(uint32_t const _divisor);
        void setGrid(uint32_t const _threads, uint32_t _blocks);
        void startChrono(std::string const& benchName);
        void stopChrono(uint32_t const index);

        bool initCleanResult(t_result** result);
        bool initCleanResult32(t_result_32** result);
        bool initCleanResult64(t_result_64** result);

#if defined(CUDA_ENABLE)
        void runNvidia();
        bool runNvidiaFnv1();
        bool runNvidiaKeccak();
        bool runNvidiaAutolykosv2();
        bool runNvidiaEthash();
        bool runNvidiaKawpow();
#endif

#if defined(AMD_ENABLE)
        void runAmd();
#endif
    };
}
