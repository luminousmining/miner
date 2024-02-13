#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <algo/algo_type.hpp>
#pragma once

#include <optional>

#include <common/cli.hpp>


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

        struct DeviceEnableSetting
        {
            bool nvidiaEnable{ true };
            bool amdEnable{ true };
            bool cpuEnable{ true };
        };

        common::Cli                    cli{};
        PoolConfig                     mining{};
        DeviceEnableSetting            deviceEnable{};
        std::vector<uint32_t>          deviceDisable{};
        std::map<uint32_t, PoolConfig> deviceSettings{};

        static Config& instance();
        bool load(int argc, char** argv);
        bool isEnable(uint32_t const deviceId) const;
        std::optional<PoolConfig> getConfigDevice(uint32_t const deviceId) const;

        algo::ALGORITHM getAlgorithm() const;
        algo::ALGORITHM getAlgorithm(std::string const& algo) const;

    private:
        bool loadCli(int argc, char** argv);
        bool isValidConfig() const;
        PoolConfig* getOrAddDeviceSettings(uint32_t const deviceId);
    };
}
