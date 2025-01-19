#pragma once

#include <benchmark/amd.hpp>
#include <benchmark/nvidia.hpp>
#include <benchmark/result.hpp>
#include <statistical/statistical.hpp>


#define RUN_BENCH(name, loopCount, _threads, _blocks, function)                \
    logInfo() << "================================";                           \
    setGrid(_threads, _blocks);                                                \
    for (uint32_t i{ 0u }; i < 10u; ++i)                                       \
    {                                                                          \
        startChrono(name);                                                     \
        if (false == (function))                                               \
        {                                                                      \
            return false;                                                      \
        }                                                                      \
        stopChrono();                                                          \
    }                                                                          \
    logInfo() << "================================";


namespace benchmark
{
    struct Benchmark
    {
    public:
        bool enableNvidia{ true };
        bool enableAmd{ true };
        benchmark::PropertiesNvidia propertiesNvidia{};
        benchmark::PropertiesAmd propertiesAmd{};

        explicit Benchmark(bool const nvidia,
                           bool const amd);
        void initializeDevices(uint32_t const deviceIndex = 0u);
        void destroyDevices();
        void run();

    private:
        std::string currentBenchName{};
        uint32_t blocks{ 1u };
        uint32_t threads{ 32u };
        uint64_t nonceComputed{ 1ull };
        statistical::Statistical stats{};

        void setGrid(uint32_t const _threads, uint32_t _blocks);
        void startChrono(std::string const& benchName);
        void stopChrono();

        bool initCleanResult(t_result** result);
        bool initCleanResult32(t_result_32** result);
        bool initCleanResult64(t_result_64** result);

        void runNvidia();
        bool runNvidiaEthash();
        bool runNvidiaAutolykosv2();
        bool runNvidiaKawpow();

        void runAmd();
    };
}
