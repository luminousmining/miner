#pragma once


#include <tuple>
#include <optional>

#include <boost/program_options.hpp>

#include <common/log/log.hpp>


namespace common
{
    struct Cli
    {
        using customParamU32 = std::tuple<uint32_t, uint32_t>;
        using customTupleU32 = std::vector<customParamU32>;

        using customParamStr = std::tuple<uint32_t, std::string>;
        using customTupleStr = std::vector<customParamStr>;

        using customParamStrStr = std::tuple<std::string, std::string>;
        using customTupleStrStr = std::vector<customParamStrStr>;

        using customParamStrStrU32 = std::tuple<std::string, std::string, uint32_t>;
        using customTupleStrStrU32 = std::vector<customParamStrStrU32>;

        boost::program_options::variables_map params{};
        boost::program_options::options_description description{ "Options" };

        explicit Cli();

        bool parse(int argc, char** argv);
        bool contains(std::string const& key) const;

        std::vector<uint32_t> getCustomMultiParamsU32(std::string const& paramName,
                                                      std::vector<std::string>& options) const;
        customTupleU32        getCustomParamsU32(std::string const& paramName, 
                                                 std::vector<std::string>& options) const;
        customTupleStr        getCustomParamsStr(std::string const& paramName,
                                                 std::vector<std::string>& options) const;
        customTupleStrStr     getCustomParamsStrStr(std::string const& paramName,
                                                    std::vector<std::string>& options) const;
        customTupleStrStrU32  getCustomParamsStrStrU32(std::string const& paramName,
                                                       std::vector<std::string>& options) const;
        void help() const;

        // logger
        std::optional<common::TYPELOG> getLevelLog() const;
        std::optional<std::string>     getLogFilenaName() const;
        std::optional<uint32_t>        getLogIntervalHashStats() const;

        // Common
         std::optional<double> getPricekWH() const;

        // Pool Connection
        std::optional<std::string>   getHost() const;
        std::optional<std::string>   getStratumType() const;
        bool                         isSSL() const;
        bool                         isStale() const;
        bool                         isSocks5() const;
        uint32_t                     getPort() const;
        std::optional<std::string>   getAlgo() const;
        std::optional<std::string>   getWallet() const;
        std::optional<std::string>   getWorkerName() const;
        std::optional<std::string>   getPassword() const;

        // Pool Custom
        std::optional<std::string>     getRavenMinerBTCWallet() const;
        std::optional<std::string>     getRavenMinerETHWallet() const;
        std::optional<std::string>     getRavenMinerLTCWallet() const;
        std::optional<std::string>     getRavenMinerBCHWallet() const;
        std::optional<std::string>     getRavenMinerADAWallet() const;
        std::optional<std::string>     getRavenMinerDODGEWallet() const;
        std::optional<std::string>     getRavenMinerMATICWallet() const;

        // Device settings common
#if defined(CUDA_ENABLE)
        bool isNvidiaEnable() const;
#endif
#if defined(AMD_ENABLE)
        bool isAmdEnable() const;
#endif
        bool isCpuEnable() const;

        // Device settings custom
        std::vector<uint32_t> getDevicesDisable() const;
        customTupleStr getCustomHost() const;
        customTupleU32 getCustomPort() const;
        customTupleStr getCustomPassword() const;
        customTupleStr getCustomAlgorithm() const;
        customTupleStr getCustomWallet() const;
        customTupleStr getCustomWorkerName() const;

        // AMD settings
        std::optional<std::string> getAMDHost() const;
        std::optional<uint32_t>    getAMDPort() const;
        std::optional<std::string> getAMDAlgo() const;

        // NVIDIA settings
        std::optional<std::string> getNvidiaHost() const;
        std::optional<uint32_t>    getNvidiaPort() const;
        std::optional<std::string> getNvidiaAlgo() const;

        // Kernel
        uint32_t getOccupancyThreads() const;
        uint32_t getOccupancyBlocks() const;
        bool     isAutoOccupancy() const;
        uint32_t getInternalLoop() const;

        // Smart mining settings
        bool isSmartMining() const;
        customTupleStrStr getSmartMiningWallet() const;
        customTupleStrStrU32 getSmartMiningPool() const;

        // Api
        uint32_t getApiPort() const;

        // Socks proxy settings
        uint32_t getSocksPort() const;
    };
}
