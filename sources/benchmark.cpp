#include <common/log/log.hpp>
#include <benchmark/benchmark.hpp>


int main()
{
    benchmark::Benchmark bench{ true, false };

    logInfo() << "Run Benchmark";

    bench.initializeDevices();
    bench.run();

    return 0;
}
