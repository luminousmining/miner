#pragma once

#include <algo/cuckatoo/cuckatoo.hpp>
#include <algo/cuckatoo/result.hpp>
#include <cstdint>


namespace resolver
{
    namespace nvidia
    {
        namespace cuckatoo32
        {
            ////////////////////////////////////////////////////////////////////////////
            // KernelParameters holds all GPU buffers and per-job state for the
            // Cuckatoo32 lean solver.
            //
            // GPU memory layout (≈ 1.5 GB total):
            //   devEdgeBitmap  – one bit per edge  (NUM_EDGES / 32 uint32_t = 512 MB)
            //   devNodeDegree  – two bits per node  (NUM_NODES / 16 uint32_t = 1 GB)
            //   devEdgeList    – compact live-edge  (MAX_EDGES_COMPACT uint32_t ≈ 4 MB)
            //   devEdgeCount   – atomic counter for compaction (4 bytes)
            ////////////////////////////////////////////////////////////////////////////
            struct KernelParameters
            {
                // SipHash-2-4 keys derived on CPU (blake2b(blake2b(pre_pow||nonce)))
                uint64_t k0{ 0ull };
                uint64_t k1{ 0ull };
                uint64_t k2{ 0ull };
                uint64_t k3{ 0ull };

                // Job context (needed by submit())
                uint64_t nonce{ 0ull };

                // GPU buffers (allocated in cuckatoo32AllocMemory)
                uint32_t* devEdgeBitmap{ nullptr };  ///< 512 MB – one bit per edge
                uint32_t* devNodeDegree{ nullptr };  ///< 1 GB  – two bits per node
                uint32_t* devEdgeList  { nullptr };  ///< compact list of live edge indices
                uint32_t* devEdgeCount { nullptr };  ///< atomic counter used by compact kernel

                // Derived sizes (compile-time constants)
                static constexpr uint64_t EDGE_BITMAP_WORDS
                    { algo::cuckatoo::NUM_EDGES / 32ULL };          // 128 M uint32_t

                static constexpr uint64_t NODE_DEGREE_WORDS
                    { algo::cuckatoo::NUM_NODES / 16ULL };          // 256 M uint32_t

                static constexpr uint32_t MAX_EDGES_COMPACT
                    { 1u << 20u };                                  // 1M entries (safe upper bound)
            };
        }
    }
}
