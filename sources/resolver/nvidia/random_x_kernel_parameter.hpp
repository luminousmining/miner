#pragma once

#include <algo/hash.hpp>
#include <algo/random_x/result.hpp>


namespace resolver
{
    namespace nvidia
    {
        namespace random_x
        {
            struct KernelParameters
            {
                uint64_t*                dataset{ nullptr };     // GPU: ~2.03 GiB
                uint8_t*                 scratchpads{ nullptr }; // GPU: blocks*threads*2 MiB
                algo::random_x::Result*  resultCache{ nullptr }; // pinned host memory

                algo::hash256            hostSeedHash{};         // last seed used (dataset rebuild trigger)
                uint64_t                 hostNonce{ 0ull };
                uint32_t                 hostTarget{ 0u };
            };
        }
    }
}
