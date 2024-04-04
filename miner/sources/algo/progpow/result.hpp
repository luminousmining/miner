#pragma once

#if !defined(__LIB_CUDA)
#include <string>
#endif

namespace algo
{
    namespace progpow
    {
        constexpr uint32_t MAX_RESULT { 4u };

#if defined(__LIB_CUDA)
        struct __align__(8) Result
#else
        struct alignas(8) Result
#endif
        {
            bool     found;
            uint32_t count;
            uint64_t nonces[4];
            uint32_t hash[algo::progpow::MAX_RESULT][algo::LEN_HASH_256_WORD_32];
        };

#if !defined(__LIB_CUDA)
        struct ResultShare
        {
            std::string jobId{};
            bool        found { false };
            uint32_t    count { 0u };
            uint64_t    nonces[4] { 0ull, 0ull, 0ull, 0ull };
            uint32_t    hash[algo::progpow::MAX_RESULT][algo::LEN_HASH_256_WORD_32]
                            {
                                {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
                                {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
                                {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
                                {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}
                            };
        };
#endif
    }
}
