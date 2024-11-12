#pragma once

#include <benchmark/amd.hpp>
#include <benchmark/nvidia.hpp>
#include <statistical/statistical.hpp>


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
        statistical::Statistical stats{};

        void startChrono(std::string const& benchName);
        void stopChrono();

        void runNvidia();
        bool runNvidiaEthash();

        void runAmd();
    };
}
