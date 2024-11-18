#pragma once


#include <network/network.hpp>
#include <stratum/progpow.hpp>


namespace stratum
{
    struct StratumQuaiPOW : public stratum::StratumProgPOW
    {
    public:
        StratumQuaiPOW();
        ~StratumQuaiPOW() = default;
    };
}
