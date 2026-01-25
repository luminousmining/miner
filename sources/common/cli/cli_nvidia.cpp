#include <common/cli/cli.hpp>


std::optional<std::string> common::Cli::getNvidiaHost() const
{
    if (true == contains("nvidia_host"))
    {
        return params["nvidia_host"].as<std::string>();
    }
    return std::nullopt;
}


std::optional<uint32_t> common::Cli::getNvidiaPort() const
{
    if (true == contains("nvidia_port"))
    {
        return params["nvidia_port"].as<uint32_t>();
    }
    return std::nullopt;
}


std::optional<std::string> common::Cli::getNvidiaAlgo() const
{
    if (true == contains("nvidia_algo"))
    {
        return params["nvidia_algo"].as<std::string>();
    }
    return std::nullopt;
}
