#include <common/cli/cli.hpp>


std::optional<double> common::Cli::getPricekWH() const
{
    if (true == contains("price_kwh"))
    {
        return params["price_kwh"].as<double>();
    }
    return std::nullopt;
}


bool common::Cli::isEthashBuildLightCacheCPU() const
{
    if (true == contains("ethash_light_cache_cpu"))
    {
        return params["ethash_light_cache_cpu"].as<bool>();
    }
    return true;
}
