#pragma once

#if !defined(__LIB_CUDA)

#if defined(_WIN32)
// Windows (MSVC and MinGW/clang): no <endian.h>. _byteswap_* live in <stdlib.h>;
// byte-order macros are provided by the compiler or defined below.
#include <stdlib.h>
#elif defined(__linux__)
#include <endian.h>
#endif

#include <cstdint>
#include <limits>
#include <string>

#include <algo/hash.hpp>
#include <common/log/log.hpp>


#if defined(_WIN32) && !defined(__BYTE_ORDER__)
constexpr std::int32_t __ORDER_LITTLE_ENDIAN__{ 1234 };
constexpr std::int32_t __ORDER_BIG_ENDIAN__{ 4321 };
constexpr std::int32_t __BYTE_ORDER__{ __ORDER_LITTLE_ENDIAN__ };
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
        inline uint32_t uint32(uint32_t const x)
        {
            return x;
        }
        inline uint64_t uint64(uint64_t const x)
        {
            return x;
        }
        inline uint64_t uint64(uint8_t const* const data)
        {
            uint64_t word{ 0ull };
            __builtin_memcpy(&word, data, sizeof(word));
            return le::uint64(word);
        }

        // Little-endian store of 4 u64 words into a 32-byte buffer.
        inline void store256(uint8_t* const data, uint64_t const* const words)
        {
            for (uint32_t w{ 0u }; w < 4u; ++w)
            {
                uint64_t const word{ le::uint64(words[w]) };
                __builtin_memcpy(data + w * sizeof(uint64_t), &word, sizeof(uint64_t));
            }
        }

        inline hash1024 const& uint32(algo::hash1024 const& h)
        {
            return h;
        }
        inline hash512 const& uint32(algo::hash512 const& h)
        {
            return h;
        }
        inline hash256 const& uint32(algo::hash256 const& h)
        {
            return h;
        }
    };

    namespace be
    {
        inline uint32_t uint32(uint32_t const x)
        {
            return bswap32(x);
        }
        inline uint64_t uint64(uint64_t const x)
        {
            return bswap64(x);
        }
    };
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#endif

    template<typename T>
    inline T hashXor(T const& x, T const& y)
    {
        T            hash{};
        size_t const length{ sizeof(T) / sizeof(uint64_t) };
        for (size_t i{ 0 }; i < length; ++i)
        {
            hash.word64[i] = x.word64[i] ^ y.word64[i];
        }
        return hash;
    }

    // 0-based bit position of the k-th set bit of `mask`, or 64 if fewer than k+1 bits are set.
    inline
    uint32_t nthSetBit(uint64_t const mask, uint32_t k)
    {
        for (uint32_t bit{ 0u }; bit < 64u; ++bit)
        {
            if (0ull != (mask & (1ull << bit)))
            {
                if (0u == k)
                {
                    return bit;
                }
                --k;
            }
        }
        return 64u;
    }

    // Parse a hex string ("0xFF" or "FF") into an unsigned integer of type T. Empty input, an
    // invalid character, or more digits than T can hold all yield 0. static_cast<T> (not the
    // fixed-width cast.hpp macros) is required so the one body serves every 8/16/32/64-bit T.
    template<typename T>
    inline
    T hexToDecimal(std::string const& text)
    {
        if (true == text.empty())
        {
            return T{ 0 };
        }
        std::string s{ text };
        if (s.size() > 2u && '0' == s[0] && ('x' == s[1] || 'X' == s[1]))
        {
            s = s.substr(2);
        }
        T value{ 0 };
        for (char const c : s)
        {
            T digit{ 0 };
            if (c >= '0' && c <= '9')
            {
                digit = static_cast<T>(c - '0');
            }
            else if (c >= 'a' && c <= 'f')
            {
                digit = static_cast<T>(c - 'a' + 10);
            }
            else if (c >= 'A' && c <= 'F')
            {
                digit = static_cast<T>(c - 'A' + 10);
            }
            else
            {
                return T{ 0 }; // invalid character -> treat the whole value as unset
            }
            if (value > static_cast<T>(std::numeric_limits<T>::max() >> 4))
            {
                return T{ 0 }; // more digits than T can hold -> invalid
            }
            value = static_cast<T>((value << 4) | digit);
        }
        return value;
    }
}

#endif // !__LIB_CUDA
