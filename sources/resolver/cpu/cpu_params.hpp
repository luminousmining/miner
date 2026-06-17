#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <optional>
#include <utility>

#include <common/cast.hpp>


namespace resolver
{
    namespace cpu
    {
        // Effective worker count: explicit --cpu_threads (>0) wins; else popcount(mask) when an
        // affinity mask is given; else the machine concurrency; never below 1.
        inline uint32_t
        resolveWorkerCount(
            std::optional<uint32_t> const threads,
            uint64_t const                mask,
            uint32_t const                hardwareConcurrency)
        {
            if (std::nullopt != threads && 0u < *threads)
            {
                return *threads;
            }
            if (0ull != mask)
            {
                return castU32(std::popcount(mask));
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
    }
}
