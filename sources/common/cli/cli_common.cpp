#include <common/cli/cli.hpp>


std::optional<double> common::Cli::getPricekWH() const
{
    if (true == contains("price_kwh"))
    {
        return params["price_kwh"].as<double>();
    }
    return std::nullopt;
}
