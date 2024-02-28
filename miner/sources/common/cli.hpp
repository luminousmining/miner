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

        // Pool Connection
        std::optional<std::string>   getHost() const;
        bool                         isSSL() const;
        bool                         isStale() const;
        uint32_t                     getPort() const;
        std::optional<std::string>   getAlgo() const;
        std::optional<std::string>   getWallet() const;
        std::optional<std::string>   getWorkerName() const;
        std::optional<std::string>   getPassword() const;

        // Device settings common
        bool isNvidiaEnable() const;
        bool isAmdEnable() const;
        bool isCpuEnable() const;

        // Device settings custom
        std::vector<uint32_t> getDevicesDisable() const;
        customTupleStr getCustomHost() const;
        customTupleU32 getCustomPort() const;
        customTupleStr getCustomPassword() const;
        customTupleStr getCustomAlgorithm() const;
        customTupleStr getCustomWallet() const;
        customTupleStr getCustomWorkerName() const;

        // Smart mining settings
        bool isSmartMining() const;
        customTupleStrStr getSmartMiningWallet() const;
        customTupleStrStrU32 getSmartMiningPool() const;
    };
}
