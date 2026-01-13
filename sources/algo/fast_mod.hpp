#pragma once

#include <cstdint>


struct alignas(16) FastDivisor
{
    uint32_t divisor;
    uint32_t magic;
    uint32_t shift;

    uint32_t padding;
};


FastDivisor initFastMod(uint32_t const d);
uint32_t fastMod(FastDivisor const& divisor, uint32_t const value);
