#include <sstream>

#include <boost/exception/diagnostic_information.hpp>

#include <common/cast.hpp>
#include <common/cli.hpp>
#include <common/log/log.hpp>

std::vector<std::string> optionDeviceEnable;
std::vector<std::string> optionDevicePool;
std::vector<std::string> optionDevicePort;
std::vector<std::string> optionDevicePassword;
std::vector<std::string> optionDeviceAlgorithm;
std::vector<std::string> optionDeviceWallet;
std::vector<std::string> optionDeviceWorkerName;


common::Cli::Cli()
{
    using namespace boost::program_options;

    description.add_options()
        (
            "help",
            "Help screen."
        )

        // Logger
        (
            "level_log",
            value<std::string>(),
            "[OPTIONAL] Set level of log.\n"
            "--level_log=<debug|info|error|warning>"
        )

        // Pool Connection
        (
            "host",
            value<std::string>(),
            "[MANDATORY] Hostname of the pool.\n"
            "--host=\"ethw.2miners.com\""
        )
        (
            "port",
            value<uint32_t>(),
            "[MANDATORY] Port of the pool.\n"
            "--port=2020"
        )
        (
            "wallet",
            value<std::string>(),
            "[MANDATORY] Wallet address.\n"
            "-wallet=\"WALLET\"")
        (
            "algo",
            value<std::string>(),
            "[MANDATORY] <ethash>\n"
            "--algo=\"etash\"")
        (
            "workername",
            value<std::string>(),
            "[MANDATORY] Name of the rig.\n"
            "--workername=\"MyWorkerName\""
        )
        (
            "password",
            value<std::string>(),
            "[OPTIONAL] Account password.\n"
            "--password=\"MyPassword\""
        )
        (
            "ssl",
            value<bool>(),
            "[OPTIONAL] Enable or not the SSL.\n"
            "Default value is false.\n"
            "--ssl=<true|false>."
        )
        (
            "stale",
            value<bool>(),
            "[OPTIONAL] Enable stale share.\n"
            "Default value is false.\n"
            "--stale=<true|false>"
        )

        // Device settings common
        (
            "nvidia",
            value<bool>(),
            "[OPTIONAL] Enable or disable device nvidia.\n"
            "Default value is true.\n"
            "--nvidia=<true|false>"
        )
        (
            "amd",
            value<bool>(),
            "[OPTIONAL] Enable or disable device amd.\n"
            "Default value is true.\n"
            "--amd=<true|false>"
        )
        (
            "cpu",
            value<bool>(),
            "[OPTIONAL] Enable or disable device cpu.\n"
            "Default value is false.\n"
            "--cpu=<true|false>"
        )

        // Device setting custom
        (
            "devices_disable",
            value<std::vector<std::string>>(&optionDeviceEnable)->multitoken(),
            "[OPTIONAL] List device disable.\n"
            "--device_disable=0,1"
        )
        (
            "device_pool",
            value<std::vector<std::string>>(&optionDevicePool)->multitoken(),
            "[OPTIONAL] Define hostname pool for custom device.\n"
            "--device_pool=0:ethw.2miners.com"
        )
        (
            "device_port",
            value<std::vector<std::string>>(&optionDevicePort)->multitoken(),
            "[OPTIONAL] Define port for custom device.\n"
            "--device_pool=0:2020"
        )
        (
            "device_password",
            value<std::vector<std::string>>(&optionDevicePassword)->multitoken(),
            "[OPTIONAL] Define password for custom device.\n"
            "--device_password=0:MyPassword"
        )
        (
            "device_algo",
            value<std::vector<std::string>>(&optionDeviceAlgorithm)->multitoken(),
            "[OPTIONAL] Define algorithm for custom device.\n"
            "--device_pool=0:kawpow"
        )
        (
            "device_wallet",
            value<std::vector<std::string>>(&optionDeviceWallet)->multitoken(),
            "[OPTIONAL] Define wallet for custom device.\n"
            "--device_pool=0:WALLET"
        )
        (
            "device_workername",
            value<std::vector<std::string>>(&optionDeviceWorkerName)->multitoken(),
            "[OPTIONAL] Define workername for custom device.\n"
            "--device_workername=0:MyWorkerName"
        )
        ;
}


bool common::Cli::parse(
    int argc,
    char** argv)
{
    try
    {
        using namespace boost::program_options;

        command_line_parser parser{argc, argv};
        parser
            .options(description)
            .allow_unregistered()
            .style(
                command_line_style::default_style | command_line_style::allow_slash_for_short);

        parsed_options parsed_options{ parser.run() };

        store(parsed_options, params);
        notify(params);
    }
    catch(boost::exception const& e)
    {
        logErr() << diagnostic_information(e);
        return false;
    }

    return true;
}


bool common::Cli::contains(
    std::string const& key) const
{
    auto const& it{ params.find(key) };
    return (it != params.end());
}


common::Cli::customTupleU32 common::Cli::getCustomParamsU32(
    std::string const& paramName,
    std::vector<std::string>& options) const
{
    customTupleU32 values;

    if (true == contains(paramName))
    {
        for (std::string const& id_value : options)
        {
            size_t const pos{ id_value.find(':') };
            if (pos == std::string::npos)
            {
                logErr() << "missing ':' [" << id_value.c_str() << "]";
                continue;
            }

            uint32_t const index{ castU32(std::atoi(id_value.substr(0, pos).c_str())) };
            std::string const subStr { id_value.substr(pos + 1, id_value.size() - 1) };

            uint32_t value { 0u };
            if (subStr == "true" || subStr == "True" || subStr == "TRUE")
            {
                value = 1;
            }
            else if (subStr == "false" || subStr == "False" || subStr == "FALSE")
            {
                value = 0;
            }
            else
            {
                value = castU32(std::atoi(subStr.c_str()));
            }

            customParamU32 customParams{ index, value };
            values.emplace_back(customParams);
        }
    }
    return values;
}


std::vector<uint32_t> common::Cli::getCustomMultiParamsU32(
    std::string const& paramName,
    std::vector<std::string>& options) const
{
    std::vector<uint32_t> values{};

    if (true == contains(paramName))
    {
        for (std::string const& id_value : options)
        {
            std::string line{};
            std::istringstream input{};
            while (std::getline(input, line, ';'))
            {
                uint32_t const value { castU32(std::atoi(line.c_str())) };
                values.emplace_back(value);
            }
        }
    }

    return values;
}


common::Cli::customTupleStr common::Cli::getCustomParamsStr(
    std::string const& paramName,
    std::vector<std::string>& options) const
{
    customTupleStr values;

    if (true == contains(paramName))
    {
        for (std::string const& id_value : options)
        {
            size_t const pos{ id_value.find(':') };
            if (pos == std::string::npos)
            {
                logErr() << "missing ':' [" << id_value.c_str() << "]";
                continue;
            }

            uint32_t const index{ castU32(std::atoi(id_value.substr(0, pos).c_str())) };
            std::string const subStr { id_value.substr(pos + 1, id_value.size() - 1) };

            customParamStr customParams{ index, subStr };
            values.emplace_back(customParams);
        }
    }
    return values;
}


void common::Cli::help() const
{
    logCustom() << description;
}
