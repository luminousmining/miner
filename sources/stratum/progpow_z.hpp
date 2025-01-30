#pragma once


#include <network/network.hpp>
#include <stratum/progpow.hpp>


namespace stratum
{
    struct StratumProgpowZ : public stratum::StratumProgPOW
    {
    public:
        StratumProgpowZ();
        ~StratumProgpowZ() = default;
    };
}
