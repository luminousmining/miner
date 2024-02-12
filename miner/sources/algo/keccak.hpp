#pragma once

#include <stdlib.h>

#include <algo/hash.hpp>


namespace algo
{
    void keccak(
        uint64_t* const out,
        uint32_t bits,
        uint8_t const* data,
        uint32_t size);

    template<typename T>
    inline
    T keccak(T const& src)
    {
        T hash{};
        keccak(
            hash.word64,
            size_t(sizeof(T) * 8),
            src.ubytes,
            sizeof(T));
        return hash;
    }

    template<typename T, typename U>
    inline
    T keccak(U const& src)
    {
        T hash{};
        keccak(
            hash.word64,
            size_t(sizeof(T) * 8),
            src.ubytes,
            sizeof(U));
        return hash;
    }

    algo::hash256 keccak(uint32_t* const src);
    algo::hash256 keccak(algo::hash800& src);
}

