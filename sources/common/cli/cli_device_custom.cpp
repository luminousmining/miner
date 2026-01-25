#include <common/cli/cli.hpp>


extern std::vector<std::string> optionDeviceEnable;
extern std::vector<std::string> optionDevicePool;
extern std::vector<std::string> optionDevicePort;
extern std::vector<std::string> optionDevicePassword;
extern std::vector<std::string> optionDeviceAlgorithm;
extern std::vector<std::string> optionDeviceWallet;
extern std::vector<std::string> optionDeviceWorkerName;


std::vector<uint32_t> common::Cli::getDevicesDisable() const
{
    return getCustomMultiParamsU32("devices_disable", optionDeviceEnable);
}


common::Cli::customTupleStr common::Cli::getCustomHost() const
{
    return getCustomParamsStr("device_pool", optionDevicePool);
}


common::Cli::customTupleU32 common::Cli::getCustomPort() const
{
    return getCustomParamsU32("device_port", optionDevicePort);
}


common::Cli::customTupleStr common::Cli::getCustomPassword() const
{
    return getCustomParamsStr("device_password", optionDevicePassword);
}


common::Cli::customTupleStr common::Cli::getCustomAlgorithm() const
{
    return getCustomParamsStr("device_algo", optionDeviceAlgorithm);
}


common::Cli::customTupleStr common::Cli::getCustomWallet() const
{
    return getCustomParamsStr("device_wallet", optionDeviceWallet);
}


common::Cli::customTupleStr common::Cli::getCustomWorkerName() const
{
    return getCustomParamsStr("device_workername", optionDeviceWorkerName);
}
