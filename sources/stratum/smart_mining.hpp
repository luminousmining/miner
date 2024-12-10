#pragma once

#include <functional>

#include <boost/thread/mutex.hpp>
#include <boost/json.hpp>

#include <algo/algo_type.hpp>
#include <network/network.hpp>
#include <stratum/job_info.hpp>
#include <stratum/stratum.hpp>


namespace stratum
{
    struct StratumSmartMining : network::NetworkTCPClient
    {
        static constexpr uint32_t ID_MINING_SUBSCRIBE { 1u };
        static constexpr uint32_t ID_SMART_MINING_SET_PROFILE { 2u };

        using callbackUpdateJob = std::function<void(stratum::StratumJobInfo const&)>;
        using callbackSetAlgorithm = std::function<void(algo::ALGORITHM const)>;
        using callbackShareStatus = std::function<void(bool const isValid,
                                                       uint32_t const requestID,
                                                       uint32_t const stratumUUID)>;
        std::string             workerName{};
        std::string             password{};
        stratum::StratumJobInfo jobInfo{};

        void setCallbackSetAlgorithm(callbackSetAlgorithm callback);
        void setCallbackUpdateJob(callbackUpdateJob callback);
        void setCallbackShareStatus(callbackShareStatus callback);

        void onConnect() final;
        void onReceive(std::string const& message) final;

        void onMethod(boost::json::object const& root);
        void onResponse(boost::json::object const& root);
        void miningSubmit(uint32_t const deviceId,
                          boost::json::array const& params);
        void miningSubmit(uint32_t const deviceId,
                          boost::json::object const& params);

        bool onSmartMiningSetAlgo(boost::json::object const& root);
        bool onSmartMiningSetExtraNonce(boost::json::object const& root);
        bool onMiningNotify(boost::json::object const& root);
        bool onMiningSetDifficulty(boost::json::object const& root);
        bool onMiningSetTarget(boost::json::object const& root);

    private:
        boost::mutex         mtxSubmit;
        callbackSetAlgorithm doSetAlgorithm{ nullptr };
        callbackUpdateJob    doUpdateJob{ nullptr };
        callbackShareStatus  doShareStatus{ nullptr };

        algo::ALGORITHM   currentAlgorithm { algo::ALGORITHM::UNKNOW };
        stratum::Stratum* stratumPool{ nullptr };

        void subscribe();
        void setProfile();
    };
}
