#pragma once


#include <network/network.hpp>
#include <stratum/stratum.hpp>


namespace stratum
{
    class StratumBlake3 : public stratum::Stratum
    {
    public:
        StratumBlake3();
        ~StratumBlake3() = default;

        void onResponse(boost::json::object const& root) final;
        void onMiningNotify(boost::json::object const& root) final;
        void onMiningSetDifficulty(boost::json::object const& root) final;
        void onMiningSetTarget(boost::json::object const& root) final;
        void onMiningSetExtraNonce(boost::json::object const& root) final;
        void onUnknowMethod(boost::json::object const& root) final;

        void miningSubscribe();
        void miningSubmit(uint32_t const deviceId,
                          boost::json::object const& params) final;
    };
}
