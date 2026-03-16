#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include <benchmark/config.hpp>
#include <benchmark/workflow.hpp>
#include <common/log/log.hpp>


int main(int argc, char** argv)
{
    ////////////////////////////////////////////////////////////////////////////
    namespace po = boost::program_options;

    ////////////////////////////////////////////////////////////////////////////
    std::string configPath{ "benchmark.json" };

    ////////////////////////////////////////////////////////////////////////////
    po::options_description desc{ "Benchmark options" };
    desc.add_options()
        ("help,h",   "Show help message")
        ("config,c", po::value<std::string>(&configPath), "Path to benchmark config JSON file (default: benchmark.json)");

    po::positional_options_description pos{};
    pos.add("config", 1);

    ////////////////////////////////////////////////////////////////////////////
    po::variables_map vm{};
    try
    {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
        po::notify(vm);
    }
    catch (std::exception const& e)
    {
        std::cerr << "Error: " << e.what() << "\n" << desc << "\n";
        return 1;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 0;
    }

    ////////////////////////////////////////////////////////////////////////////
    benchmark::Config const config{ benchmark::Config::loadFromFile(configPath) };

    benchmark::BenchmarkWorkflow bench{ config };

    logInfo() << "Run Workflow Benchmarks";

    if (false == bench.initializeDevices())
    {
        return 1;
    }
    bench.run();

    return 0;
}
