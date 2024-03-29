#pragma once

#if !defined(__LIB_CUDA)
#include <string>
#endif

namespace algo
{
    namespace autolykos_v2
    {
#if defined(__LIB_CUDA)
        struct __align__(8) Result
#else
        struct alignas(8) Result
#endif
        {
            bool     found;
            uint32_t count;
            uint64_t nonces[4];
        };

#if !defined(__LIB_CUDA)
        struct ResultShare
        {
            std::string jobId{};
            bool        found{ false };
            uint32_t    count{ 0u };
            uint32_t    extraNonceSize{ 0u };
            uint32_t    extraNonce2Size{ 0u };
            uint64_t    nonces[4]{ 0ull, 0ull, 0ull, 0ull };
        };
#endif
    }
}
