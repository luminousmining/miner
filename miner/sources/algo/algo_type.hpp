#pragma once


#include <string>


namespace algo
{
    enum class ALGORITHM : uint8_t
    {
        SHA256,
        ETHASH,
        ETCHASH,
        KAWPOW,
        FIROPOW,
        AUTOLYKOS_V2,
        UNKNOW,
        MAX_SIZE = algo::ALGORITHM::UNKNOW
    };

    std::string toString(algo::ALGORITHM const algorithm);
}
