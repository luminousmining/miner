#pragma once

#if !defined(__LIB_CUDA)

#if defined(_MSC_VER)
#include <stdlib.h>
#elif defined(__linux__) || defined(__GNUC__)
#include <endian.h>
#endif

#include <algo/hash.hpp>
#include <common/log/log.hpp>


#if defined(_WIN32) && !defined(__BYTE_ORDER__)
constexpr std::int32_t __ORDER_LITTLE_ENDIAN__ { 1234 };
constexpr std::int32_t __ORDER_BIG_ENDIAN__    { 4321 };
constexpr std::int32_t __BYTE_ORDER__          { __ORDER_LITTLE_ENDIAN__ };
#elif (defined(__linux__) || defined(__GNUC__)) && !defined(__BYTE_ORDER__)
#error "__GNUC__ should define __BYTE_ORDER__"
#endif

#if defined(__GNUC__)
#define bswap32 __builtin_bswap32
#define bswap64 __builtin_bswap64
#elif defined(_MSC_VER)
#include <string.h>
#define __builtin_memcpy memcpy
#define bswap32 _byteswap_ulong
#define bswap64 _byteswap_uint64
#endif


namespace algo
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    namespace le
    {
        inline uint32_t uint32(uint32_t const x) { return x; }
        inline uint64_t uint64(uint64_t const x) { return x; }
        inline uint64_t uint64(uint8_t const* const data)
        {
            uint64_t word{ 0ull };
            __builtin_memcpy(&word, data, sizeof(word));
            return le::uint64(word);
        }

        inline hash1024 const& uint32(algo::hash1024 const& h) { return h; }
        inline hash512  const& uint32(algo::hash512 const& h)  { return h; }
        inline hash256  const& uint32(algo::hash256 const& h)  { return h; }
    };

    namespace be
    {
        inline uint32_t uint32(uint32_t const x) { return bswap32(x); }
        inline uint64_t uint64(uint64_t const x) { return bswap64(x); }
    };
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#endif

    template<typename T>
    inline
    T hashXor(T const& x, T const& y)
    {
        T hash{};
        size_t const length { sizeof(T) / sizeof(uint64_t) };
        for (size_t i{ 0 }; i < length; ++i)
        {
            hash.word64[i] = x.word64[i] ^ y.word64[i];
        }
        return hash;
    }
}

#endif // !__LIB_CUDA
