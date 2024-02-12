#pragma once

#if !defined(__LIB_CUDA)
#include <string>
#endif

namespace algo
{
    namespace progpow
    {
#if defined(__LIB_CUDA)
        struct __align__(16) Result
#else
        struct alignas(16) Result
#endif
        {
            bool     found;
            uint32_t count;
            uint64_t nonces[4];
            uint32_t hash[4][8];
        };

#if !defined(__LIB_CUDA)
        struct ResultShare
        {
            std::string jobId{};
            bool        found { false };
            uint32_t    count { 0u };
            uint64_t    nonces[4] { 0ull, 0ull, 0ull, 0ull };
            uint32_t    hash[4][8] { {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
                                    {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
                                    {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
                                    {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u} };
        };
#endif
    }
}
