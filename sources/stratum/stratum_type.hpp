#pragma once

#include <cstdint>
#include <string>


namespace stratum
{
    enum class STRATUM_TYPE : uint8_t
    {
        // TODO: V1, -> basic stratum bitcoin
        ETHEREUM_V1,
        ETHEREUM_V2,
        ETHPROXY,
    };

    enum class ETHEREUM_V2_ID : uint32_t
    {
        MINING_HELLO = 1u,
        MINING_AUTHORIZE = 2u,
        MINING_NOOP = 7u
    };

    enum class ETHPROXY_ID : uint32_t
    {
        EMPTY = 0u,
        SUBMITLOGIN = 1u,
        GETWORK = 5u
    };

    std::string toString(stratum::STRATUM_TYPE const stratumType);
}
