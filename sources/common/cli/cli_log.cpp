#include <common/cli/cli.hpp>


std::optional<common::TYPELOG> common::Cli::getLevelLog() const
{
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


std::optional<std::string> common::Cli::getLogFilenaName() const
{
    if (true == contains("log_file"))
    {
        return params["log_file"].as<std::string>();
    }

    return std::nullopt;
}


std::optional<uint32_t> common::Cli::getLogIntervalHashStats() const
{
    if (true == contains("log_interval_hash"))
    {
        return params["log_interval_hash"].as<uint32_t>();
    }

    return std::nullopt;
}


bool common::Cli::isLogNewJob() const
{
    if (true == contains("log_new_job"))
    {
        return params["log_new_job"].as<bool>();
    }

    return true;
}
