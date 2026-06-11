#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>


namespace resolver::cpu_detail
{
    // 0-based bit position of the k-th set bit of mask, or 64 if fewer than k+1 bits set.
    inline uint32_t nthSetBit(uint64_t const mask, uint32_t k)
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

    // Effective worker count: explicit --cpu_threads (>0) wins; else popcount(mask) when an
    // affinity mask is given; else the machine concurrency; never below 1.
    inline uint32_t
    resolveWorkerCount(std::optional<uint32_t> const threads, uint64_t const mask, uint32_t const hardwareConcurrency)
    {
        if (threads.has_value() && 0u < *threads)
        {
            return *threads;
        }
        if (0ull != mask)
        {
            return static_cast<uint32_t>(std::popcount(mask));
        }
        return (0u < hardwareConcurrency) ? hardwareConcurrency : 1u;
    }

    // Contiguous half-open sub-range [lo, hi) for worker `index` of `workerCount`, splitting
    // [0, total). The first (total % workerCount) workers take one extra item so coverage is
    // exact with no gaps or overlaps.
    inline std::pair<uint64_t, uint64_t>
    chunkRange(uint64_t const total, uint32_t const workerCount, uint32_t const index)
    {
        uint64_t const base{ total / workerCount };
        uint64_t const rem{ total % workerCount };
        uint64_t const lo{ index * base + std::min<uint64_t>(index, rem) };
        uint64_t const extra{ (index < rem) ? 1ull : 0ull };
        return { lo, lo + base + extra };
    }

    // Parse a hex affinity mask ("0xFF" or "FF") to uint64. Empty or invalid input -> 0.
    inline uint64_t parseHexMask(std::string const& text)
    {
        if (true == text.empty())
        {
            return 0ull;
        }
        std::string s{ text };
        if (s.size() > 2u && '0' == s[0] && ('x' == s[1] || 'X' == s[1]))
        {
            s = s.substr(2);
        }
        uint64_t value{ 0ull };
        for (char const c : s)
        {
            uint64_t digit{ 0ull };
            if (c >= '0' && c <= '9')
            {
                digit = static_cast<uint64_t>(c - '0');
            }
            else if (c >= 'a' && c <= 'f')
            {
                digit = static_cast<uint64_t>(c - 'a' + 10);
            }
            else if (c >= 'A' && c <= 'F')
            {
                digit = static_cast<uint64_t>(c - 'A' + 10);
            }
            else
            {
                return 0ull; // invalid character -> treat the whole mask as unset
            }
            if (value > (std::numeric_limits<uint64_t>::max() >> 4))
            {
                return 0ull; // more than 64 bits of mask -> invalid (documented <=64 cores)
            }
            value = (value << 4) | digit;
        }
        return value;
    }
}
