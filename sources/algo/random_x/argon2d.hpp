#pragma once

#include <cstdint>


namespace algo
{
    namespace random_x
    {
        void buildCache(uint8_t* cache, uint8_t const* key, uint32_t keyLen);

        // Convenience overload for Monero's fixed 32-byte seed hash
        inline void buildCache(uint8_t* cache, uint8_t const* seedHash)
        {
            buildCache(cache, seedHash, 32u);
        }
    }
}
