#pragma once


#include <cstdint>


namespace algo
{
    constexpr uint32_t FNV1_PRIME { 0x01000193u };
    constexpr uint32_t FNV1_OFFSET { 0x811c9dc5u };

    uint32_t fnv1(uint32_t const u, uint32_t const v);
    uint32_t fnv1a(uint32_t const u, uint32_t const v);
}
