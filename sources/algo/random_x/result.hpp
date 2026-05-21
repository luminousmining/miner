#pragma once

#if !defined(__LIB_CUDA)
#include <string>
#endif


namespace algo
{
    namespace random_x
    {
        constexpr uint32_t MAX_RESULT{ 4u };

        struct alignas(8) Result
        {
            bool     found{ false };
            uint32_t count{ 0u };
            uint32_t nonces[algo::random_x::MAX_RESULT]{};
            uint8_t  hash[32]{};
        };

#if !defined(__LIB_CUDA)
        struct ResultShare
        {
            std::string jobId{};
            bool        found{ false };
            uint32_t    count{ 0u };
            uint32_t    nonces[algo::random_x::MAX_RESULT]{};
            uint8_t     hashes[algo::random_x::MAX_RESULT][32]{};
        };
#endif
    }
}
