#pragma once

#include <functional>

#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
#include <boost/json.hpp>

#include <algo/algo_type.hpp>
#include <network/network.hpp>
#include <stratum/job_info.hpp>
#include <stratum/stratum_type.hpp>


namespace stratum
{
    struct Stratum : public network::NetworkTCPClient
    {
    public:
        // Stratum
        static constexpr uint32_t OVERCOM_NONCE{ 1000u };
        static constexpr uint32_t ID_MINING_SUBSCRIBE{ 1u };
        static constexpr uint32_t ID_MINING_AUTHORIZE{ 2u };
        static constexpr uint32_t ID_MINING_SUBMIT{ OVERCOM_NONCE };

        using callbackUpdateJob = std::function<void(uint32_t const stratumUUID,
                                                     StratumJobInfo const&)>;
        using callbackShareStatus = std::function<void(bool const isValid,
                                                       uint32_t const requestID,
                                                       uint32_t const stratumUUID)>;

        uint32_t              uuid{ 0u };
        StratumJobInfo        jobInfo{};
        algo::ALGORITHM       algorithm{ algo::ALGORITHM::UNKNOW };
        stratum::STRATUM_TYPE stratumType{ stratum::STRATUM_TYPE::STRATUM };
        std::string           workerName{};
        std::string           wallet{};
        std::string           password{};
        std::string           stratumName{};

        // Stratum EthereumV2

        std::string workerID{};
        uint32_t maxErrors{ 0u };
        uint32_t resume{ 0u };
        uint32_t timeout{ 0u };

        virtual ~Stratum();
        virtual void onResponse(boost::json::object const& root) = 0;
        virtual void miningSubmit(uint32_t const deviceId,
                                  boost::json::array const& params);
        virtual void miningSubmit(uint32_t const deviceId,
                                  boost::json::object const& params);

        virtual void onMiningSet(boost::json::object const& root);
        virtual void onMiningSetTarget(boost::json::object const& root);
        virtual void onMiningSetExtraNonce(boost::json::object const& root);
        virtual void onMiningNotify(boost::json::object const& root) = 0;
        virtual void onMiningSetDifficulty(boost::json::object const& root) = 0;

        virtual void onConnect();
        virtual void onUnknowMethod(boost::json::object const& root);
        virtual void updateJob();
        virtual void miningHello();
        virtual void miningSubscribe();
        virtual void miningAuthorize();

        void onReceive(std::string const& message) final;

        void setCallbackUpdateJob(callbackUpdateJob cbUpdateJob);
        void setCallbackShareStatus(callbackShareStatus cbShareStatus);
        void onMethod(boost::json::object const& root);
        void setExtraNonce(std::string const& paramExtraNonce);
        void setExtraNonce(std::string const& paramExtraNonce,
                           uint32_t const paramExtraNonce2Size);
        void doLoopTimeout();
        bool isValidJob() const;

    protected:
        bool                authenticated{ false };
        boost::mutex        mtxSubmit;
        boost::mutex        mtxDispatchJob{};
        boost::thread       threadTimeout{};
        callbackUpdateJob   dispatchJob{ nullptr };
        callbackShareStatus shareStatus{ nullptr };

        void onShare(boost::json::object const& root,
                     uint32_t const miningRequestID);
        void loopTimeout();
    };
}