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
    };
}
