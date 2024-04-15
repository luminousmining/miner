#pragma once


#include <algo/blake3/result.hpp>
#include <algo/hash.hpp>


namespace resolver
{
    namespace nvidia
    {
        namespace blake3
        {
            struct KernelParameters
            {
                uint64_t       hostNonce { 0ull };
                uint32_t       hostFromGroup { 0u };
                uint32_t       hostToGroup { 0u };
                algo::hash256  hostBoundary {};
                algo::hash256  hostTargetBlob {};
                algo::hash3072 hostHeaderBlob {};

                algo::hash3072*       header { nullptr };
                algo::hash256*        target { nullptr };
                algo::blake3::Result* resultCache { nullptr };
            };
        }
    }
}
