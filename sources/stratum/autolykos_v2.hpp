#pragma once


#include <network/network.hpp>
#include <stratum/stratum.hpp>


namespace stratum
{
    class StratumAutolykosV2 : public stratum::Stratum
    {
    public:
        void onResponse(boost::json::object const& root) final;
        void onMiningNotify(boost::json::object const& root) final;
        void onMiningSetDifficulty(boost::json::object const& root) final;
        void onMiningSetTarget(boost::json::object const& root) final;
        void onMiningSetExtraNonce(boost::json::object const& root) final;

        void miningSubmit(uint32_t const deviceId,
                          boost::json::array const& params) final;
    };
}
