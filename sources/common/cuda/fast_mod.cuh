#pragma once


#include <algo/fast_mod.hpp>


__device__ __forceinline__
uint32_t fast_mod(
    FastDivisor const divisor,
    uint32_t const value)
{
    uint32_t const d{ divisor.divisor };
    uint32_t const m{ divisor.magic };
    uint32_t const s{ divisor.shift };

    uint32_t q{ __umulhi(value, m) >> s };

    uint32_t r{ value - (q * d) };

    while (r >= d)
    {
        r -= d;
    }

    return r;
}
