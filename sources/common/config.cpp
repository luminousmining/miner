#include <algo/math.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/env_utils.hpp>
#include <common/log/log.hpp>
#include <common/log/log_file.hpp>


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

    if (true == cli.contains("help"))
    {
        cli.help();
        return false;
    }

    return isValidConfig();
}


bool common::Config::isValidConfig() const
{
    bool error{ false };

    if (common::PROFILE::STANDARD == profile)
    {
        CLI_CHECK(mining.host.empty() == true,       "missing --host");
        CLI_CHECK(mining.port == 0,                  "missing --port");
        CLI_CHECK(mining.algo.empty() == true,       "missing --algo");
        CLI_CHECK(mining.workerName.empty() == true, "missing --workername");
        CLI_CHECK(mining.password.empty() == true,   "missing --password");
        CLI_CHECK(mining.wallet.empty() == true,     "missing --wallet");
    }
    else if (common::PROFILE::SMART_MINING == profile)
    {
        CLI_CHECK(mining.workerName.empty() == true,          "missing --workername");
        CLI_CHECK(mining.password.empty() == true,            "missing --password");
        CLI_CHECK(smartMining.coinPoolConfig.empty() == true, "missing --sm_wallet and --sm_pool");
        for (auto const& it : smartMining.coinPoolConfig)
        {
            PoolConfig const& poolConfig { it.second };
            CLI_CHECK(poolConfig.host.empty() == true,     "missing --sm_pool");
            CLI_CHECK(poolConfig.port == 0,                "missing --sm_pool");
            CLI_CHECK(poolConfig.wallet.empty() == true,   "missing --sm_wallet");
            CLI_CHECK(poolConfig.password.empty() == true, "missing --sm_wallet");
        }
    }

    return !error;
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
        // LOGGER
        ////////////////////////////////////////////////////////////////////////
        auto const levelLog { cli.getLevelLog() };
        if (std::nullopt != levelLog)
        {
            log.level = *levelLog;
            setLogLevel(*levelLog);
        }

        auto const logFile { cli.getLogFilenaName() };
        if (std::nullopt != logFile)
        {
            log.file.assign(*logFile);
            common::LoggerFile::instance().openFilename();
        }

        auto const intervalHashStats { cli.getLogIntervalHashStats() };
        if (std::nullopt != intervalHashStats)
        {
            log.intervalHashStats = common::min_limit(*intervalHashStats, 100u);
        }

        log.showNewJob = cli.isLogNewJob();

        ////////////////////////////////////////////////////////////////////////
        // ENVIRONMENT
        ////////////////////////////////////////////////////////////////////////
        auto const cudaLazy{ cli.getEnvironmentCudaLazy() };
        if (std::nullopt != cudaLazy)
        {
            common::setVarEnv
            (
                "CUDA_MODULE_LOADING",
                *cudaLazy == true ? "LAZY" : "EAGER"
            );
        }

        auto const cudaDeviceOrder{ cli.getEnvironmentCudadeviceOrder() };
        if (std::nullopt != cudaDeviceOrder)
        {
            std::string const order = *cudaDeviceOrder;
            if ("PCI_BUS_ID" == order || "FASTEST_FIRST" == order)
            {
                common::setVarEnv("CUDA_DEVICE_ORDER", order.c_str());
            }
        }

        auto const gpuHeapSize{ cli.getEnvironmentGpuHeapSize() };
        if (std::nullopt != gpuHeapSize)
        {
            common::setVarEnv("GPU_MAX_HEAP_SIZE", common::min_limit(*gpuHeapSize, 10u));
        }

        auto const gpuMaxAllocPercent{ cli.getEnvironmentGpuMaxAllocPercent() };
        if (std::nullopt != gpuMaxAllocPercent)
        {
            common::setVarEnv("GPU_MAX_ALLOC_PERCENT", common::min_limit(*gpuMaxAllocPercent, 10u));
        }

        auto const gpuSingleAllocPercent{ cli.getEnvironmentGpuSingleAllocPercent() };
        if (std::nullopt != gpuMaxAllocPercent)
        {
            common::setVarEnv("GPU_SINGLE_ALLOC_PERCENT", common::min_limit(*gpuSingleAllocPercent, 10u));
        }

        ////////////////////////////////////////////////////////////////////////
        // COMMON
        ////////////////////////////////////////////////////////////////////////
        auto const priceKWH{ cli.getPricekWH() };
        if (std::nullopt != priceKWH)
        {
            common.priceKWH = *priceKWH;
        }

        ////////////////////////////////////////////////////////////////////////
        // POOL
        ////////////////////////////////////////////////////////////////////////
        auto const host{ cli.getHost() };
        if (std::nullopt != host && false == host->empty())
        {
            mining.host.assign(*host);
        }

        auto const stratumType{ cli.getStratumType() };
        if (std::nullopt != stratumType && false == stratumType->empty())
        {
            if ("v1" == stratumType)
            {
                mining.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V1;
            }
            else if ("v2" == stratumType)
            {
                mining.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V2;
            }
            else if ("ethproxy" == stratumType)
            {
                mining.stratumType = stratum::STRATUM_TYPE::ETHPROXY;
            }
            // TODO: implement stratum BITCOIN
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

        mining.secrureConnect = cli.isSSL();
        mining.stale = cli.isStale();
        mining.socks5 = cli.isSocks5();
        if (true == mining.socks5)
        {
            auto const SocksPort { cli.getSocksPort() };
            if (true == algo::inRange(1u, 65535u, SocksPort))
            {
                mining.socksPort = SocksPort;
            }
        }

        auto const socksIp{ cli.getSocksIp() };

        mining.socksIp = socksIp;
        

        ////////////////////////////////////////////////////////////////////////
        // RAVEN MINER
        ////////////////////////////////////////////////////////////////////////
        auto const ravenMinerBTCWallet{ cli.getRavenMinerBTCWallet() };
        if (std::nullopt != ravenMinerBTCWallet)
        {
            mining.wallet.assign(*ravenMinerBTCWallet);
            mining.host.assign("stratum.ravenminer.com");
            mining.port = 3838u;
            mining.password = "btc";
            mining.algo = "kawpow";
        }
        auto const ravenMinerETHWallet{ cli.getRavenMinerETHWallet() };
        if (std::nullopt != ravenMinerETHWallet)
        {
            mining.wallet.assign(*ravenMinerETHWallet);
            mining.host.assign("stratum.ravenminer.com");
            mining.port = 3838u;
            mining.password = "eth";
            mining.algo = "kawpow";
        }
        auto const ravenMinerLTCWallet{ cli.getRavenMinerLTCWallet() };
        if (std::nullopt != ravenMinerLTCWallet)
        {
            mining.wallet.assign(*ravenMinerLTCWallet);
            mining.host.assign("stratum.ravenminer.com");
            mining.port = 3838u;
            mining.password = "ltc";
            mining.algo = "kawpow";
        }
        auto const ravenMinerBCHWallet{ cli.getRavenMinerBCHWallet() };
        if (std::nullopt != ravenMinerBCHWallet)
        {
            mining.wallet.assign(*ravenMinerBCHWallet);
            mining.host.assign("stratum.ravenminer.com");
            mining.port = 3838u;
            mining.password = "bch";
            mining.algo = "kawpow";
        }
        auto const ravenMinerADAWallet{ cli.getRavenMinerADAWallet() };
        if (std::nullopt != ravenMinerADAWallet)
        {
            mining.wallet.assign(*ravenMinerADAWallet);
            mining.host.assign("stratum.ravenminer.com");
            mining.port = 3838u;
            mining.password = "ada";
            mining.algo = "kawpow";
        }
        auto const ravenMinerDODGEWallet{ cli.getRavenMinerDODGEWallet() };
        if (std::nullopt != ravenMinerDODGEWallet)
        {
            mining.wallet.assign(*ravenMinerDODGEWallet);
            mining.host.assign("stratum.ravenminer.com");
            mining.port = 3838u;
            mining.password = "dodge";
            mining.algo = "kawpow";
        }
        auto const ravenMinerMATICWallet{ cli.getRavenMinerMATICWallet() };
        if (std::nullopt != ravenMinerMATICWallet)
        {
            mining.wallet.assign(*ravenMinerMATICWallet);
            mining.host.assign("stratum.ravenminer.com");
            mining.port = 3838u;
            mining.password = "matic";
            mining.algo = "kawpow";
        }

        ////////////////////////////////////////////////////////////////////////
        // DEVICES SETTING CUSTOM
        ////////////////////////////////////////////////////////////////////////
#if defined(CUDA_ENABLE)
        deviceEnable.nvidiaEnable = cli.isNvidiaEnable();
#endif
#if defined(AMD_ENABLE)
        deviceEnable.amdEnable = cli.isAmdEnable();
#endif
        deviceEnable.cpuEnable = cli.isCpuEnable();

#if defined(AMD_ENABLE)
        ////////////////////////////////////////////////////////////////////////
        // DEVICE SETTING AMD
        ////////////////////////////////////////////////////////////////////////
        if (true == deviceEnable.amdEnable)
        {
            auto const amdHost{ cli.getAMDHost() };
            auto const amdPort{ cli.getAMDPort() };
            auto const amdAlgo{ cli.getAMDAlgo() };

            if (   std::nullopt != amdHost
                && std::nullopt != amdPort
                && std::nullopt != amdAlgo)
            {
                amdSetting.host.assign(*amdHost);
                amdSetting.port = *amdPort;
                amdSetting.algo = *amdAlgo;
            }
        }
#endif

        ////////////////////////////////////////////////////////////////////////
        // DEVICE SETTING NVIDIA
        ////////////////////////////////////////////////////////////////////////
#if defined(CUDA_ENABLE)
        if (true == deviceEnable.nvidiaEnable)
        {
            auto const nvidiaHost{ cli.getNvidiaHost() };
            auto const nvidiaPort{ cli.getNvidiaPort() };
            auto const nvidiaAlgo{ cli.getNvidiaAlgo() };

            if (   std::nullopt != nvidiaHost
                && std::nullopt != nvidiaPort
                && std::nullopt != nvidiaAlgo)
            {
                nvidiaSetting.host.assign(*nvidiaHost);
                nvidiaSetting.port = *nvidiaPort;
                nvidiaSetting.algo = *nvidiaAlgo;
            }
        }
#endif

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

        ////////////////////////////////////////////////////////////////////////
        // KERNEL
        ////////////////////////////////////////////////////////////////////////
        auto const isAutoOccupancy{ cli.isAutoOccupancy() };
        auto const occupancyThreads{ cli.getOccupancyThreads() };
        auto const occupancyBlocks{ cli.getOccupancyBlocks() };
        auto const internalLoop{ cli.getInternalLoop() };
        auto const kernelMinimunExecuteNeeded{ cli.getMinimunKernelExecuted() };
        auto const cudaContext{ cli.getCudaContext() };
        occupancy.isAuto = isAutoOccupancy;
        occupancy.internalLoop = internalLoop;
        occupancy.cudaContext = cudaContext;
        occupancy.kernelMinimunExecuteNeeded = kernelMinimunExecuteNeeded;
        if (   0u != occupancyThreads
            || 0u != occupancyBlocks)
        {
            if (0u != occupancyThreads)
            {
                occupancy.threads = occupancyThreads;
            }
            if (0u != occupancyBlocks)
            {
                occupancy.blocks = occupancyBlocks;
            }
            logInfo() << "Disable auto occupancy, using override occupancy!";
            occupancy.isAuto = false;
        }


        ////////////////////////////////////////////////////////////////////////
        // ALGORITHM
        ////////////////////////////////////////////////////////////////////////
        deviceAlgorithm.ethashBuildLightCacheCPU = cli.isEthashBuildLightCacheCPU();

        ////////////////////////////////////////////////////////////////////////
        // SMART MINING
        ////////////////////////////////////////////////////////////////////////
        if (true == cli.isSmartMining())
        {
            profile = common::PROFILE::SMART_MINING;

            for (auto const& it : cli.getSmartMiningWallet())
            {
                std::string const& smCoin{ std::get<0>(it) };
                std::string const& smWallet{ std::get<1>(it) };

                if (smartMining.coinPoolConfig.find(smCoin) == smartMining.coinPoolConfig.end())
                {
                    smartMining.coinPoolConfig[smCoin].wallet.assign(smWallet);
                }
                else
                {
                    logErr() << "sm_wallet => duplicate coin[" << smCoin << "]";
                    return false;
                }
            }

            for (auto const& it : cli.getSmartMiningPool())
            {
                std::string const& smartMiningCoin{ std::get<0>(it) };
                std::string const& smartMiningUrl{ std::get<1>(it) };
                uint32_t const& smartMiningPort{ std::get<2>(it) };

                common::Config::PoolConfig& poolConfig { smartMining.coinPoolConfig[smartMiningCoin] };
                if (   false == poolConfig.host.empty()
                    || 0u != poolConfig.port)
                {
                    logErr() << "sm_pool => duplicate coin[" << smartMiningCoin << "]";
                    return false;
                }
                poolConfig.host.assign(smartMiningUrl);
                poolConfig.port = smartMiningPort;
                poolConfig.password = mining.password;
            }
        }
        
        ////////////////////////////////////////////////////////////////////////
        // API
        ////////////////////////////////////////////////////////////////////////
        auto const ApiPort { cli.getApiPort() };
        if (true == algo::inRange(1u, 65535u, ApiPort))
        {
            api.port = ApiPort;
        }

        ////////////////////////////////////////////////////////////////////////
        // TOOL MOCKER
        ////////////////////////////////////////////////////////////////////////
#if defined(TOOLS_ENABLE) &&  defined(TOOL_MOCKER)
        toolConfigs.mockerResolverCount = cli.getMockerResolverCount();
        toolConfigs.mockerResolverUpdateMemorySleep = cli.getMockerResolverUpdateMemorySleep();
#endif
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
    return algo::toEnum(mining.algo);
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
