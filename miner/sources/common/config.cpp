#include <algo/math.hpp>
#include <common/config.hpp>
#include <common/log/log.hpp>


#define CLI_CHECK(condition, msg) if ((condition)) { error = true; logErr() << msg; }


common::Config& common::Config::instance()
{
    static common::Config handler;
    return handler;
}


bool common::Config::load(int argc, char** argv)
{
    if (false == loadCli(argc, argv))
    {
        logErr() << "Fail load";
        return false;
    }

    return isValidConfig();
}


bool common::Config::isValidConfig() const
{
    bool error{ false };
    CLI_CHECK(mining.host.empty() == true, "missing --host");
    CLI_CHECK(mining.port == 0, "missing --port");
    CLI_CHECK(mining.algo.empty() == true, "missing --algo");
    CLI_CHECK(mining.workerName.empty() == true, "missing --wokername");
    CLI_CHECK(mining.wallet.empty() == true, "missing --wallet");

    return true;
}


common::Config::PoolConfig* common::Config::getOrAddDeviceSettings(
    uint32_t const deviceId)
{
    if (deviceSettings.find(deviceId) == deviceSettings.end())
    {
        deviceSettings[deviceId] = {};
    }
    return &deviceSettings.at(deviceId);
}


bool common::Config::loadCli(int argc, char** argv)
{
    try
    {
        if (false == cli.parse(argc, argv))
        {
            logErr() << "Fail parse cli";
            return false;
        }

        ////////////////////////////////////////////////////////////////////////
        auto const levelLog { cli.getLevelLog() };
        if (std::nullopt != levelLog)
        {
            setLogLevel(*levelLog);
        }

        ////////////////////////////////////////////////////////////////////////
        auto const host{ cli.getHost() };
        if (std::nullopt != host && false == host->empty())
        {
            mining.host.assign(*host);
        }

        auto const port{ cli.getPort() };
        if (true == algo::inRange(1u, 65535u, port))
        {
            mining.port = port;
        }

        auto const algo{ cli.getAlgo() };
        if (std::nullopt != algo && false == algo->empty())
        {
            mining.algo.assign(*algo);
        }

        auto const wallet{ cli.getWallet() };
        if (std::nullopt != wallet && false == wallet->empty())
        {
            mining.wallet.assign(*wallet);
        }

        auto const password{ cli.getPassword() };
        if (std::nullopt != password && false == password->empty())
        {
            mining.password.assign(*password);
        }

        auto const workerName{ cli.getWorkerName() };
        if (std::nullopt != workerName && false == workerName->empty())
        {
            mining.workerName.assign(*workerName);
        }

        mining.secrureConnect = cli.getSSL();
        mining.stale = cli.getStale();

        ////////////////////////////////////////////////////////////////////////
        deviceEnable.nvidiaEnable = cli.isNvidiaEnable();
        deviceEnable.amdEnable = cli.isAmdEnable();
        deviceEnable.cpuEnable = cli.isCpuEnable();

        ////////////////////////////////////////////////////////////////////////
        for (uint32_t const& deviceIdDisable : cli.getDevicesDisable())
        {
            deviceDisable.emplace_back(deviceIdDisable);
        }
        for (auto const& it : cli.getCustomHost())
        {
            uint32_t const& index{ std::get<0>(it) };
            std::string const& value{ std::get<1>(it) };
            common::Config::PoolConfig* poolConfig{ getOrAddDeviceSettings(index) };
            poolConfig->host.assign(value);
        }
        for (auto const& it : cli.getCustomPort())
        {
            uint32_t const& index{ std::get<0>(it) };
            uint32_t const& value{ std::get<1>(it) };
            common::Config::PoolConfig* poolConfig{ getOrAddDeviceSettings(index) };
            poolConfig->port = value;
        }
        for (auto const& it : cli.getCustomPassword())
        {
            uint32_t const& index{ std::get<0>(it) };
            std::string const& value{ std::get<1>(it) };
            common::Config::PoolConfig* poolConfig{ getOrAddDeviceSettings(index) };
            poolConfig->password.assign(value);
        }
        for (auto const& it : cli.getCustomAlgorithm())
        {
            uint32_t const& index{ std::get<0>(it) };
            std::string const& value{ std::get<1>(it) };
            common::Config::PoolConfig* poolConfig{ getOrAddDeviceSettings(index) };
            poolConfig->algo.assign(value);
        }
        for (auto const& it : cli.getCustomWallet())
        {
            uint32_t const& index{ std::get<0>(it) };
            std::string const& value{ std::get<1>(it) };
            common::Config::PoolConfig* poolConfig{ getOrAddDeviceSettings(index) };
            poolConfig->wallet.assign(value);
        }
        for (auto const& it : cli.getCustomWorkerName())
        {
            uint32_t const& index{ std::get<0>(it) };
            std::string const& value{ std::get<1>(it) };
            common::Config::PoolConfig* poolConfig{ getOrAddDeviceSettings(index) };
            poolConfig->workerName.assign(value);
        }
    }
    catch(std::exception const& e)
    {
        logErr() << e.what() << '\n';
        return false;
    }
    
    return true;
}


bool common::Config::isEnable(uint32_t const deviceId) const
{
    auto const& it
    {
        std::find_if(
            deviceDisable.begin(),
            deviceDisable.end(),
            [&](uint32_t id) { return deviceId == id; })
    };

    return it == deviceDisable.end();
}


algo::ALGORITHM common::Config::getAlgorithm() const
{
    return getAlgorithm(mining.algo);
}


algo::ALGORITHM common::Config::getAlgorithm(
    std::string const& algo) const
{
    if      (algo == "sha256")      { return algo::ALGORITHM::SHA256;       }
    else if (algo == "ethash")      { return algo::ALGORITHM::ETHASH;       }
    else if (algo == "etchash")     { return algo::ALGORITHM::ETCHASH;      }
    else if (algo == "progpow")     { return algo::ALGORITHM::PROGPOW;      }
    else if (algo == "kawpow")      { return algo::ALGORITHM::KAWPOW;       }
    else if (algo == "firopow")     { return algo::ALGORITHM::FIROPOW;      }
    else if (algo == "autolykosv2") { return algo::ALGORITHM::AUTOLYKOS_V2; }

    return algo::ALGORITHM::UNKNOW;
}


std::optional<common::Config::PoolConfig> common::Config::getConfigDevice(
    uint32_t const deviceId) const
{
    auto it { deviceSettings.find(deviceId) };
    if (it != deviceSettings.end())
    {
        return { it->second };
    }
    return std::nullopt;
}
