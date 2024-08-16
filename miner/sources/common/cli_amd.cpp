#include <common/cli.hpp>


std::optional<std::string> common::Cli::getAMDHost() const
{
    if (true == contains("amd_host"))
    {
        return params["amd_host"].as<std::string>();
    }
    return std::nullopt;
}


std::optional<uint32_t> common::Cli::getAMDPort() const
{
    if (true == contains("amd_port"))
    {
        return params["amd_port"].as<uint32_t>();
    }
    return std::nullopt;
}


std::optional<std::string> common::Cli::getAMDAlgo() const
{
    if (true == contains("amd_algo"))
    {
        return params["amd_algo"].as<std::string>();
    }
    return std::nullopt;
}
