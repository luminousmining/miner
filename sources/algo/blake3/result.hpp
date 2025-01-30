#pragma once

#if !defined(__LIB_CUDA)
#include <string>
#endif


namespace algo
{
    namespace blake3
    {
        constexpr uint32_t MAX_RESULT { 4u };

        struct alignas(8) Result
        {
            bool     found;
            uint32_t count;
            uint64_t nonces[algo::blake3::MAX_RESULT];
        };

#if !defined(__LIB_CUDA)
        struct ResultShare
        {
            std::string jobId{};
            uint32_t    toGroup{ 0u };
            uint32_t    fromGroup{ 0u };
            bool        found{ false };
            uint32_t    extraNonceSize { 0u };
            uint32_t    count{ 0u };
            uint64_t    nonces[algo::blake3::MAX_RESULT]{ 0ull, 0ull, 0ull, 0ull };
        };
#endif
    }
}
