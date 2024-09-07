#include <common/cli.hpp>


std::optional<std::string> common::Cli::getRavenMinerBTCWallet() const
{
    if (true == contains("rm_rvn_btc"))
    {
        return params["rm_rvn_btc"].as<std::string>();
    }

    return std::nullopt;
}


std::optional<std::string> common::Cli::getRavenMinerETHWallet() const
{
    if (true == contains("rm_rvn_eth"))
    {
        return params["rm_rvn_eth"].as<std::string>();
    }

    return std::nullopt;
}


std::optional<std::string> common::Cli::getRavenMinerLTCWallet() const
{
    if (true == contains("rm_rvn_ltc"))
    {
        return params["rm_rvn_ltc"].as<std::string>();
    }

    return std::nullopt;
}


std::optional<std::string> common::Cli::getRavenMinerLTCWallet() const
{
    if (true == contains("rm_rvn_bch"))
    {
        return params["rm_rvn_bch"].as<std::string>();
    }

    return std::nullopt;
}


std::optional<std::string> common::Cli::getRavenMinerBWallet() const
{
    if (true == contains("rm_rvn_ada"))
    {
        return params["rm_rvn_ada"].as<std::string>();
    }

    return std::nullopt;
}


std::optional<std::string> common::Cli::getRavenMinerBWallet() const
{
    if (true == contains("rm_rvn_dodge"))
    {
        return params["rm_rvn_dodge"].as<std::string>();
    }

    return std::nullopt;
}


std::optional<std::string> common::Cli::getRavenMinerBWallet() const
{
    if (true == contains("rm_rvn_matic"))
    {
        return params["rm_rvn_matic"].as<std::string>();
    }

    return std::nullopt;
}
