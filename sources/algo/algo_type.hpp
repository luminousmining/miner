#pragma once

#include <cstdint>
#include <string>


namespace algo
{
    enum class ALGORITHM : uint8_t
    {
        SHA256,
        ETHASH,
        ETCHASH,
        PROGPOW,
        PROGPOWQUAI,
        PROGPOWZ,
        KAWPOW,
        MEOWPOW,
        FIROPOW,
        EVRPROGPOW,
        AUTOLYKOS_V2,
        BLAKE3,
        UNKNOWN,
        MAX_SIZE = algo::ALGORITHM::UNKNOWN
    };

    std::string toString(algo::ALGORITHM const algorithm);
    algo::ALGORITHM toEnum(std::string const& algorithm);
}
