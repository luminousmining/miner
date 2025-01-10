#pragma once


#include <network/network.hpp>
#include <stratum/stratum.hpp>


namespace stratum
{
    struct StratumSha256 : public stratum::Stratum
    {
    public:
        void onResponse(boost::json::object const& root) final;
        void onMiningNotify(boost::json::object const& root) final;
        void onMiningSetDifficulty(boost::json::object const& root) final;

        void miningSubscribe() final;
        void miningSubmit(uint32_t const deviceId,
                          boost::json::array const& params) final;

    private:
        std::string sessionId{};
        uint32_t extraNonce2Size{ 0u };
    };
}
