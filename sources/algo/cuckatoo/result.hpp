#pragma once

#include <algo/cuckatoo/cuckatoo.hpp>

#if !defined(__LIB_CUDA)
#include <string>
#endif


namespace algo
{
    namespace cuckatoo
    {
        ////////////////////////////////////////////////////////////////////////
        // GPU-side result (written by the cycle-detection kernel)
        struct alignas(4) Result
        {
            bool     found;
            uint32_t proof[algo::cuckatoo::PROOF_SIZE]; // 42 sorted nonces
        };

        ////////////////////////////////////////////////////////////////////////
        // Host-side share (copied from GPU result, enriched with job context)
#if !defined(__LIB_CUDA)
        struct ResultShare
        {
            std::string jobId{};
            uint64_t    height{ 0ull };
            uint32_t    grinJobId{ 0u };
            uint64_t    nonce{ 0ull };
            bool        found{ false };
            uint32_t    proof[algo::cuckatoo::PROOF_SIZE]{};
        };
#endif
    }
}
