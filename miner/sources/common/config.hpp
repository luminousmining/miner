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
#if defined(CUDA_ENABLE)
            bool nvidiaEnable{ true };
#endif
            bool amdEnable{ true };
            bool cpuEnable{ true };
        };

        struct LogConfig
        {
            common::TYPELOG level { common::TYPELOG::__INFO };
            std::string file{};
        };

        common::PROFILE                profile { common::PROFILE::STANDARD };
        common::Cli                    cli{};
        LogConfig                      log{};
        PoolConfig                     mining{};
        SmartMiningConfig              smartMining{};
        DeviceEnableSetting            deviceEnable{};
        std::vector<uint32_t>          deviceDisable{};
        std::map<uint32_t, PoolConfig> deviceSettings{};
        PoolConfig                     amdSetting{};
        PoolConfig                     nvidiaSetting{};

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
