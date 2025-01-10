#pragma once


#include <network/network.hpp>
#include <stratum/progpow.hpp>


namespace stratum
{
    struct StratumProgpowQuai : public stratum::StratumProgPOW
    {
    public:
        StratumProgpowQuai();
        ~StratumProgpowQuai() = default;

        void onResponse(boost::json::object const& root) final;
        void onMiningSet(boost::json::object const& root) final;
        void onMiningNotify(boost::json::object const& root) final;
    };
}
