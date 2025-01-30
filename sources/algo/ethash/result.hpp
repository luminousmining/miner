#pragma once

#if !defined(__LIB_CUDA)
#include <string>
#endif

namespace algo
{
    namespace ethash
    {
        constexpr uint32_t MAX_RESULT { 4u };

        struct alignas(8) Result
        {
            bool     found{ false };
            uint32_t count{ 0u };
            uint64_t nonces[algo::ethash::MAX_RESULT]{ 0u, 0u, 0u, 0u };
        };

#if !defined(__LIB_CUDA)
        struct ResultShare
        {
            std::string jobId{};
            bool        found { false };
            uint32_t    extraNonceSize { 0u };
            uint32_t    count { 0u };
            uint64_t    nonces[algo::ethash::MAX_RESULT] { 0ull, 0ull, 0ull, 0ull };
        };
#endif
    }
}
