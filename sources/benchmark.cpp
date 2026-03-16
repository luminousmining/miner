#include <benchmark/workflow.hpp>
#include <common/log/log.hpp>


int main()
{
    benchmark::BenchmarkWorkflow bench{ true, true };

    logInfo() << "Run Workflow Benchmarks";

    if (false == bench.initializeDevices())
    {
        return 1;
    }
    bench.run();

    return 0;
}
