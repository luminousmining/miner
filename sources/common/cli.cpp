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
std::vector<std::string> optionSmartMiningWallet;
std::vector<std::string> optionSmartMiningPool;


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
        (
            "log_file",
            value<std::string>(),
            "[OPTIONAL] Set path to write log.\n"
            "--log_file=PATH"
        )

        // Common
        (
            "price_kwh",
            value<double>(),
            "[OPTIONAL] Set the price of elec (kWh).\n"
            "--price_kwh=0.5"
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
            "-wallet=\"WALLET\""
        )
        (
            "algo",
            value<std::string>(),
            "[MANDATORY] <ethash>\n"
            "--algo=\"ethash\""
        )
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
        (
            "socks5",
            value<bool>(),
            "[OPTIONAL] Enable SOCKS5 proxy.\n"
            "Default value is false.\n"
            "--socks5=<true|false>."
        )
        (
            "socks_port",
            value<uint32_t>(),
            "[OPTIONAL] Port of the SOCKS proxy.\n"
            "--socks_port=9050"
        )

        // Pool Custom
        (
            "rm_rvn_btc",
            value<std::string>(),
            "[OPTIONAL] Mining on ravenminer RVN with BTC wallet\n"
            "--rm_rvn_btc=WALLET"
        )
        (
            "rm_rvn_eth",
            value<std::string>(),
            "[OPTIONAL] Mining on ravenminer RVN with ETH wallet\n"
            "--rm_rvn_eth=WALLET"
        )
        (
            "rm_rvn_ltc",
            value<std::string>(),
            "[OPTIONAL] Mining on ravenminer RVN with LTC wallet\n"
            "--rm_rvn_ltc=WALLET"
        )
        (
            "rm_rvn_bch",
            value<std::string>(),
            "[OPTIONAL] Mining on ravenminer RVN with BCH wallet\n"
            "--rm_rvn_bch=WALLET"
        )
        (
            "rm_rvn_ada",
            value<std::string>(),
            "[OPTIONAL] Mining on ravenminer RVN with ADA wallet\n"
            "--rm_rvn_ada=WALLET"
        )
        (
            "rm_rvn_dodge",
            value<std::string>(),
            "[OPTIONAL] Mining on ravenminer RVN with DODGE wallet\n"
            "--rm_rvn_dodge=WALLET"
        )
        (
            "rm_rvn_matic",
            value<std::string>(),
            "[OPTIONAL] Mining on ravenminer RVN with MATIC wallet\n"
            "--rm_rvn_matic=WALLET"
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

        // AMD setting
        (
            "amd_host",
            value<bool>(),
            "[OPTIONAL] Set defaut hostname of pool for all gpu AMD.\n"
            "If defined, the parameters amd_port and amd_algo must be defined.\n"
            "--amd_host=\"ethw.2miners.com\""
        )
        (
            "amd_port",
            value<uint32_t>(),
            "[OPTIONAL] Set port of the pool for all gpu AMD.\n"
            "If defined, the parameters amd_host and amd_algo must be defined.\n"
            "--amd_port=2020"
        )
        (
            "amd_algo",
            value<std::string>(),
            "[MANDATORY] <ethash>\n"
            "If defined, the parameters amd_host and amd_port must be defined.\n"
            "--amd_algo=\"ethash\""
        )

        // NVIDIA setting
        (
            "nvidia_host",
            value<bool>(),
            "[OPTIONAL] Set defaut hostname of pool for all gpu NVIDIA.\n"
            "If defined, the parameters nvidia_port and nvidia_algo must be defined.\n"
            "--nvidia_host=\"ethw.2miners.com\""
        )
        (
            "nvidia_port",
            value<uint32_t>(),
            "[OPTIONAL] Set port of the pool for all gpu NVIDIA.\n"
            "If defined, the parameters nvidia_host and nvidia_algo must be defined.\n"
            "--nvidia_port=2020"
        )
        (
            "nvidia_algo",
            value<std::string>(),
            "[MANDATORY] <ethash>\n"
            "If defined, the parameters nvidia_host and nvidia_port must be defined.\n"
            "--nvidia_algo=\"ethash\""
        )

        // Device setting custom
        (
            "devices_disable",
            value<std::vector<std::string>>(&optionDeviceEnable)->multitoken(),
            "[OPTIONAL] List device disable.\n"
            "--devices_disable=0,1"
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

        // Kernel
        (
            "threads",
            value<uint32_t>(),
            "[OPTIONAL] Set occupancy threads.\n"
            "--threads=128"
        )
        (
            "blocks",
            value<uint32_t>(),
            "[OPTIONAL] Set occupancy blocks.\n"
            "--blocks=128"
        )
        (
            "occupancy",
            value<bool>(),
            "[OPTIONAL] System will define the best occupancy for kernel.\n"
            "--occupancy=true|false"
        )

        // smart mining
        (
            "sm_wallet",
            value<std::vector<std::string>>(&optionSmartMiningWallet)->multitoken(),
            "[OPTIONAL] assign a wallet with a coin.\n"
            "--sm_wallet=COIN:WALLET"
        )
        (
            "sm_pool",
            value<std::vector<std::string>>(&optionSmartMiningPool)->multitoken(),
            "[OPTIONAL] assign a pool with a coin.\n"
            "--sm_pool=COIN@POOL_URL:POOL_PORT"
        )

        // api setting
        (
            "api_port",
            value<uint32_t>(),
            "[OPTIONAL] Set port of the api.\n"
            "--api_port=8080"
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
            if (std::string::npos == pos)
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
        size_t pos { 0u };

        for (std::string flags : options)
        {
            uint32_t index{ 0u };
            pos = flags.find(',');
            while (pos != std::string::npos)
            {
                index = castU32(std::atoi(flags.substr(0, pos).c_str()));
                flags.erase(0, pos + 1);
                values.emplace_back(index);
                pos = flags.find(',');
            }
            index = castU32(std::atoi(flags.substr(0, pos).c_str()));
            values.emplace_back(index);
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
            if (std::string::npos == pos)
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


common::Cli::customTupleStrStr common::Cli::getCustomParamsStrStr(
    std::string const& paramName,
    std::vector<std::string>& options) const
{
    customTupleStrStr values;

    if (true == contains(paramName))
    {
        for (std::string const& id_value : options)
        {
            size_t const pos{ id_value.find(':') };
            if (std::string::npos == pos)
            {
                logErr() << "missing ':' [" << id_value.c_str() << "]";
                continue;
            }

            std::string const right{ id_value.substr(0, pos) };
            std::string const left{ id_value.substr(pos + 1, id_value.size() - 1) };

            customParamStrStr customParams{ right, left };
            values.emplace_back(customParams);
        }
    }

    return values;
}


common::Cli::customTupleStrStrU32 common::Cli::getCustomParamsStrStrU32(
    std::string const& paramName,
    std::vector<std::string>& options) const
{
    customTupleStrStrU32 values;

    if (true == contains(paramName))
    {
        for (std::string const& id_value : options)
        {
            size_t pos{ id_value.find('@') };
            if (std::string::npos == pos)
            {
                logErr() << "missing '@' [" << id_value.c_str() << "]";
                continue;
            }

            std::string const right{ id_value.substr(0, pos) };
            std::string const left{ id_value.substr(pos + 1, id_value.size() - 1) };

            pos = left.find(':');
            if (std::string::npos == pos)
            {
                logErr() << "missing ':' [" << left.c_str() << "]";
                continue;
            }
            std::string const leftLeft { left.substr(0, pos) };
            std::string const leftRight { left.substr(pos + 1, left.size() - 1) };
            uint32_t const valueLeftRight { castU32(std::atoi(leftRight.c_str())) };

            customParamStrStrU32 customParams { right, leftLeft, valueLeftRight };
            values.emplace_back(customParams);
        }
    }

    return values;
}


void common::Cli::help() const
{
    logCustom() << description;
}
