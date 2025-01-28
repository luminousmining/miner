#pragma once

#include <string>


namespace stratum
{
    enum class STRATUM_TYPE : uint8_t
    {
        ETHEREUM_V1,
        ETHEREUM_V2,
        ETHPROXY
    };

    enum class ETHEREUM_V2_ID : uint32_t
    {
        MINING_HELLO = 1,
        MINING_AUTHORIZE = 2,
        MINING_NOOP = 7
    };

    std::string toString(stratum::STRATUM_TYPE const stratumType);
}
