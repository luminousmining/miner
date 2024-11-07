#pragma once

#include <benchmark/amd.hpp>
#include <benchmark/nvidia.hpp>


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
        void runNvidia();
        void runNvidiaEthash();

        void runAmd();
    };
}
