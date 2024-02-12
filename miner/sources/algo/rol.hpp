#pragma once


namespace algo
{
    inline
    uint64_t rol_u64(
        uint64_t const x,
        uint32_t const s)
    {
        return (x << s) | (x >> (64 - s));
    }


    inline
    uint32_t rol_u32(
        uint32_t const x,
        uint32_t const n)
    {
        return (x << n) | (x >> (32 - n));
    }


    inline
    uint32_t ror_u32(
        uint32_t const x,
        uint32_t const n)
    {
        return (x >> n) | (x << (32 - n));
    }
}
