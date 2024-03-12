#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>

#include <algo/algo_type.hpp>
#pragma once

#include <optional>

#include <common/cli.hpp>
#include <common/profile.hpp>


namespace common
{
    struct Config
    {
    public:
        struct PoolConfig
        {
            std::string   host{ "" };
            uint32_t      port{ 0u };
            uint32_t      retryConnectionCount{ 10 };
            std::string   algo{ "" };
            std::string   workerName{ "" };
            std::string   wallet{ "" };
            std::string   password{ "x" };
            bool          secrureConnect{ false };
            bool          stale { false };
        };

        struct SmartMiningConfig
        {
            using MapCoinPool = std::unordered_map<std::string/*coin*/, PoolConfig>;
            MapCoinPool coinPoolConfig{};
        };

        struct DeviceEnableSetting
        {
            bool nvidiaEnable{ true };
            bool amdEnable{ true };
            bool cpuEnable{ true };
        };

        struct DeviceOptionsDev
        {
            bool doubleStream { false };
        };

        common::PROFILE                      profile { common::PROFILE::STANDARD };
        common::Cli                          cli{};
        PoolConfig                           mining{};
        SmartMiningConfig                    smartMining{};
        DeviceEnableSetting                  deviceEnable{};
        std::vector<uint32_t>                deviceDisable{};
        std::map<uint32_t, PoolConfig>       deviceSettings{};
        std::map<uint32_t, DeviceOptionsDev> deviceOptionDev{};

        static Config& instance();
        bool load(int argc, char** argv);
        bool isEnable(uint32_t const deviceId) const;
        std::optional<PoolConfig> getConfigDevice(uint32_t const deviceId) const;
        std::optional<DeviceOptionsDev> getConfigDev(uint32_t const deviceId) const;

        algo::ALGORITHM getAlgorithm() const;

    private:
        bool loadCli(int argc, char** argv);
        bool isValidConfig() const;
        PoolConfig* getOrAddDeviceSettings(uint32_t const deviceId);
        DeviceOptionsDev* getOrAddDevOption(uint32_t const deviceId);
    };
}
