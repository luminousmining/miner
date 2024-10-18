#pragma once


#include <algo/hash.hpp>

namespace algo
{
    template<typename T>
    struct HashContext
    {
        uint64_t numberItem{ 0ull };
        uint64_t size{ 0ull };
        T*       hash{ nullptr };
    };

    struct DagContext
    {
        char*                       data{ nullptr };
        int32_t                     epoch{ -1 };
        uint64_t                    period{ 0ull };
        HashContext<algo::hash512>  lightCache{};
        HashContext<algo::hash1024> dagCache{};
    };
}