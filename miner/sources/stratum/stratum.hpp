#pragma once


#include <functional>

#include <boost/thread/mutex.hpp>
#include <boost/json.hpp>

#include <algo/algo_type.hpp>
#include <network/network.hpp>
#include <stratum/job_info.hpp>


namespace stratum
{
    constexpr char STRATUM_VERSION[22]{ "EthereumStratum/1.0.0" };

    struct Stratum : public network::NetworkTCPClient
    {
    public:
        static constexpr uint32_t OVERCOM_NONCE { 1000u };
        static constexpr uint32_t ID_MINING_SUBSCRIBE { 1u };
        static constexpr uint32_t ID_MINING_AUTHORIZE { 2u };
        static constexpr uint32_t ID_MINING_SUBMIT { OVERCOM_NONCE };

        using callbackUpdateJob = std::function<void(uint32_t const _stratumUUID,
                                                     StratumJobInfo const&)>;
        using callbackShareStatus = std::function<void(bool const isValid,
                                                       uint32_t const requestID,
                                                       uint32_t const _stratumUUID)>;

        uint32_t        uuid { 0u };
        StratumJobInfo  jobInfo{};
        algo::ALGORITHM algorithm { algo::ALGORITHM::UNKNOW };
        std::string     workerName{};
        std::string     wallet{};
        std::string     password{};

        Stratum();
        virtual ~Stratum();

        virtual void onResponse(boost::json::object const& root) = 0;
        virtual void miningSubmit(uint32_t const deviceId,
                                  boost::json::array const& params) = 0;

        virtual void onMiningNotify(boost::json::object const& root) = 0;
        virtual void onMiningSetDifficulty(boost::json::object const& root) = 0;
        virtual void onMiningSetTarget(boost::json::object const& root) = 0;
        virtual void onMiningSetExtraNonce(boost::json::object const& root) = 0;

        virtual void onConnect();
        virtual void updateJob();
        virtual void miningSubscribe();
        virtual void miningAuthorize();

        void onReceive(std::string const& message) final;

        void setCallbackUpdateJob(callbackUpdateJob cbUpdateJob);
        void setCallbackShareStatus(callbackShareStatus cbShareStatus);
        void onMethod(boost::json::object const& root);
        void setExtraNonce(std::string const& paramExtraNonce);
        void setExtraNonce(std::string const& paramExtraNonce,
                           uint32_t const paramExtraNonce2Size);
        bool isValidJob() const;

    protected:
        bool                authenticated{ false };
        boost::mutex        mtxSubmit;
        boost::mutex        mtxDispatchJob{};
        callbackUpdateJob   dispatchJob{};
        callbackShareStatus shareStatus{};

        void onShare(boost::json::object const& root,
                     uint32_t const miningRequestID);
    };
}