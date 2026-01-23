#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

#include <algo/algo_type.hpp>
#include <device/type.hpp>
#include <common/cli.hpp>
#include <common/profile.hpp>
#include <stratum/stratum_type.hpp>


namespace common
{
    struct Config
    {
    public:
        struct CommonConfig
        {
            double priceKWH{ 0.0 };
        };

        struct PoolConfig
        {
            std::string           host{ "" };
            stratum::STRATUM_TYPE stratumType{ stratum::STRATUM_TYPE::ETHEREUM_V1 };
            uint32_t              port{ 0u };
            uint32_t              retryConnectionCount{ 10 };
            uint32_t              socksPort{ 0u };
            std::string           algo{ "" };
            std::string           workerName{ "" };
            std::string           wallet{ "" };
            std::string           password{ "x" };
            bool                  secrureConnect{ false };
            bool                  stale{ false };
            bool                  socks5{ false };
        };

        struct SmartMiningConfig
        {
            using MapCoinPool = std::unordered_map<std::string/*coin*/, PoolConfig>;
            MapCoinPool coinPoolConfig{};
        };

        struct DeviceOccupancy
        {
            bool                    isAuto{ false };
            std::optional<uint32_t> threads{};
            std::optional<uint32_t> blocks{};
            std::optional<uint32_t> internalLoop{};
        };

        struct DeviceEnableSetting
        {
#if defined(CUDA_ENABLE)
            bool nvidiaEnable{ true };
#endif
#if defined(AMD_ENABLE)
            bool amdEnable{ true };
#endif
            bool cpuEnable{ true };
        };

        struct LogConfig
        {
            common::TYPELOG level { common::TYPELOG::__INFO };
            std::string file{};
        };

        struct ApiConfig
        {
            uint32_t port{ 8080u };
        };

        common::PROFILE                profile { common::PROFILE::STANDARD };
        common::Cli                    cli{};
        LogConfig                      log{};
        PoolConfig                     mining{};
        DeviceOccupancy                occupancy{};
        SmartMiningConfig              smartMining{};
        DeviceEnableSetting            deviceEnable{};
        std::vector<uint32_t>          deviceDisable{};
        std::map<uint32_t, PoolConfig> deviceSettings{};
        PoolConfig                     amdSetting{};
        PoolConfig                     nvidiaSetting{};
        ApiConfig                      api{};
        CommonConfig                   common{};

        static Config& instance();
        bool load(int argc, char** argv);
        bool isEnable(uint32_t const deviceId) const;
        std::optional<PoolConfig> getConfigDevice(uint32_t const deviceId) const;

        algo::ALGORITHM getAlgorithm() const;

    private:
        bool loadCli(int argc, char** argv);
        bool isValidConfig() const;
        PoolConfig* getOrAddDeviceSettings(uint32_t const deviceId);
    };
}
