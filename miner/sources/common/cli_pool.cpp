#include <common/cli.hpp>


std::optional<common::TYPELOG> common::Cli::getLevelLog() const
{
    std::optional<std::string> value;
    if (true == contains("level_log"))
    {
        std::string levelLog{ params["level_log"].as<std::string>() };
        if (levelLog == "debug")
        {
            return common::TYPELOG::__DEBUG;
        }
        else if (levelLog == "info")
        {
            return common::TYPELOG::__INFO;
        }
        else if (levelLog == "warning")
        {
            return common::TYPELOG::__WARNING;
        }
        else if (levelLog == "error")
        {
            return common::TYPELOG::__ERROR;
        }
    }
    return std::nullopt;
}


std::optional<std::string> common::Cli::getHost() const
{
    std::optional<std::string> value;
    if (true == contains("host"))
    {
        return params["host"].as<std::string>();
    }
    return std::nullopt;
}


bool common::Cli::getSSL() const
{
    if (true == contains("ssl"))
    {
        return params["ssl"].as<bool>();
    }
    return false;
}


bool common::Cli::getStale() const
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
    std::optional<std::string> value;
    if (true == contains("algo"))
    {
        return params["algo"].as<std::string>();
    }
    return std::nullopt;
}


std::optional<std::string> common::Cli::getWallet() const
{
    std::optional<std::string> value;
    if (true == contains("wallet"))
    {
        return params["wallet"].as<std::string>();
    }
    return std::nullopt;
}


std::optional<std::string> common::Cli::getWorkerName() const
{
    std::optional<std::string> value;
    if (true == contains("workername"))
    {
        return params["workername"].as<std::string>();
    }
    return std::nullopt;
}


std::optional<std::string> common::Cli::getPassword() const
{
    std::optional<std::string> value;
    if (true == contains("password"))
    {
        return params["password"].as<std::string>();
    }
    return std::nullopt;
}
