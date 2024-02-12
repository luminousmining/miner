#pragma once


#include <network/network.hpp>
#include <stratum/progpow.hpp>


namespace stratum
{
    struct StratumFiroPOW : public stratum::StratumProgPOW
    {
    public:
        StratumFiroPOW();
        ~StratumFiroPOW() = default;
    };
}
