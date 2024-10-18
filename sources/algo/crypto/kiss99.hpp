#pragma once


#include <cstdint>

namespace algo
{
    struct Kiss99Properties
    {
        uint32_t z{ 0u };
        uint32_t w{ 0u };
        uint32_t jsr{ 0u };
        uint32_t jcong{ 0u };
    };

    uint32_t kiss99(algo::Kiss99Properties& data);
}
