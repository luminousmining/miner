#pragma once

#include <string>


namespace algo
{
    enum class ALGORITHM : uint8_t
    {
        SHA256,
        ETHASH,
        ETCHASH,
        ETHASH_BLAKE3,
        PROGPOW,
        PROGPOWQUAI,
        PROGPOWZ,
        KAWPOW,
        MEOWPOW,
        FIROPOW,
        EVRPROGPOW,
        AUTOLYKOS_V2,
        BLAKE3,
        UNKNOW,
        MAX_SIZE = algo::ALGORITHM::UNKNOW
    };

    std::string toString(algo::ALGORITHM const algorithm);
    algo::ALGORITHM toEnum(std::string const& algorithm);
}
