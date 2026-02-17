#include <common/cli/cli.hpp>


std::optional<std::string> common::Cli::getHost() const
{
    if (true == contains("host"))
    {
        return params["host"].as<std::string>();
    }
    return std::nullopt;
}


std::optional<std::string> common::Cli::getStratumType() const
{
    if (true == contains("stratum"))
    {
        return params["stratum"].as<std::string>();
    }
    return std::nullopt;
}


bool common::Cli::isSSL() const
{
    if (true == contains("ssl"))
    {
        return params["ssl"].as<bool>();
    }
    return false;
}


bool common::Cli::isSocks5() const
{
    if (true == contains("socks5"))
    {
        return params["socks5"].as<bool>();
    }
    return false;
}


bool common::Cli::isStale() const
{
    if (true == contains("stale"))
    {
        return params["stale"].as<bool>();
    }
    return false;
}


uint32_t common::Cli::getPort() const
{
    if (true == contains("port"))
    {
        return params["port"].as<uint32_t>();
    }
    return 0u;
}


std::optional<std::string> common::Cli::getAlgo() const
{
    if (true == contains("algo"))
    {
        return params["algo"].as<std::string>();
    }
    return std::nullopt;
}


std::optional<std::string> common::Cli::getWallet() const
{
    if (true == contains("wallet"))
    {
        return params["wallet"].as<std::string>();
    }
    return std::nullopt;
}


std::optional<std::string> common::Cli::getWorkerName() const
{
    if (true == contains("workername"))
    {
        return params["workername"].as<std::string>();
    }
    return std::nullopt;
}


std::optional<std::string> common::Cli::getPassword() const
{
    if (true == contains("password"))
    {
        return params["password"].as<std::string>();
    }
    return std::nullopt;
}


uint32_t common::Cli::getApiPort() const
{
    if (true == contains("api_port"))
    {
        return params["api_port"].as<uint32_t>();
    }
    return 8080u;
}


uint32_t common::Cli::getSocksPort() const
{
    if (true == contains("socks_port"))
    {
        return params["socks_port"].as<uint32_t>();
    }
    return 9050u;
}

std::optional<std::string> common::Cli::getSocksHost() const
{
    if (true == contains("socks_host"))
    {
        return params["socks_host"].as<std::string>();
    }
    return "127.0.0.1";
}