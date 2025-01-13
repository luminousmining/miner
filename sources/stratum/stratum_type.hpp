#pragma once


namespace stratum
{
    enum STRATUM_TYPE : uint8_t
    {
        STRATUM,
        ETHEREUM_V2
    };

    enum ETHEREUM_V2_ID : uint32_t
    {
        MINING_HELLO,
        MINING_SUBSCRIBE,
        MINING_AUTHORIZE
    };
}
