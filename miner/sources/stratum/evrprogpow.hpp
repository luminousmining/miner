#pragma once


#include <network/network.hpp>
#include <stratum/progpow.hpp>


namespace stratum
{
    struct StratumEvrprogPOW : public stratum::StratumProgPOW
    {
    public:
        StratumEvrprogPOW();
        ~StratumEvrprogPOW() = default;
    };
}
