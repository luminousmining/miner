#pragma once


#include <network/network.hpp>
#include <stratum/progpow.hpp>


namespace stratum
{
    struct StratumMeowPOW : public stratum::StratumProgPOW
    {
    public:
        StratumMeowPOW();
        ~StratumMeowPOW() = default;
    };
}
