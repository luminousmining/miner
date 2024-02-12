#pragma once


#include <network/network.hpp>
#include <stratum/ethash.hpp>


namespace stratum
{
    struct StratumEtchash : public stratum::StratumEthash
    {
    public:
        StratumEtchash();
        ~StratumEtchash() = default;
    };
}
