#pragma once

#include <algo/hash.hpp>
#include <algo/kheavyhash/result.hpp>


namespace resolver
{
    namespace nvidia
    {
        namespace kheavyhash
        {
            // kHeavyHash is not memory-hard: per-job state is the host-generated
            // 64x64 nibble matrix, the 32-byte pre-pow header and 32-byte LE target.
            struct KernelParameters
            {
                uint64_t      hostNonce{ 0ull };
                uint64_t      hostTimestamp{ 0ull };
                algo::hash256 hostHeader{};
                algo::hash256 hostTarget{};
                uint16_t      hostMatrix[64u * 64u]{};

                uint16_t*                 matrix{ nullptr };       // device, 4096 x u16
                algo::hash256*            header{ nullptr };       // device
                algo::hash256*            target{ nullptr };       // device
                algo::kheavyhash::Result* resultCache{ nullptr };  // pinned host, [2] (double-buffered)
            };
        }
    }
}
