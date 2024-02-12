#pragma once


#include <algo/autolykos/result.hpp>
#include <algo/hash.hpp>


namespace resolver
{
    namespace nvidia
    {
        namespace autolykos_v2
        {
            struct KernelParameters
            {
                uint32_t hostPeriod { 0u };
                uint32_t hostHeight { 0u };
                uint32_t hostDagItemCount { 0u };
                uint64_t hostNonce { 0ull };
                algo::hash256 hostBoundary {};
                algo::hash256 hostHeader {};

                algo::hash256* header { nullptr };
                algo::hash256* dag { nullptr };
                algo::hash256* BHashes { nullptr };
                algo::autolykos_v2::Result* resultCache { nullptr };
            };
        }
    }
}
