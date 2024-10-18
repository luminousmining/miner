#pragma once


#include <network/network.hpp>
#include <stratum/progpow.hpp>


namespace stratum
{
    struct StratumKawPOW : public stratum::StratumProgPOW
    {
    public:
        StratumKawPOW();
        ~StratumKawPOW() = default;
    };
}
