////////////////////////////////////////////////////////////////////////////////
// Cuckatoo32 lean solver – CUDA implementation
//
// Algorithm:
//   1. Seed:    initialise edge bitmap to all-ones (all 2^32 edges alive)
//   2. Trim:    repeat TRIM_ROUNDS times for each side (U then V):
//                a. zero degree-counter array
//                b. count degrees of live edges into 2-bit counters
//                c. kill edges whose endpoint has degree < 2
//   3. Compact: collect surviving edge indices into a dense array
//   4. Cycle:   DFS on CPU to find the 42-cycle (proof)
//
// GPU memory per device (≈1.5 GB):
//   devEdgeBitmap  uint32_t[NUM_EDGES/32]   512 MB  one bit per edge
//   devNodeDegree  uint32_t[NUM_NODES/16]     1 GB  2-bit saturated counter per node
//   devEdgeList    uint32_t[MAX_EDGES_COMPACT] ~4 MB compact live-edge indices
//   devEdgeCount   uint32_t[1]                tiny   atomic counter for compact kernel
////////////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cuda_runtime.h>

#include <algo/cuckatoo/cuckatoo.hpp>
#include <algo/cuckatoo/result.hpp>
#include <common/error/cuda_error.hpp>
#include <common/log/log.hpp>
#include <resolver/nvidia/cuckatoo32_kernel_parameter.hpp>

#include <algo/cuckatoo/cuda/siphash.cuh>
#include <algo/cuckatoo/cuda/cuckatoo32.cuh>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>


////////////////////////////////////////////////////////////////////////////////
// GPU constant memory – SipHash keys (set once per job via cudaMemcpyToSymbol)
////////////////////////////////////////////////////////////////////////////////
__constant__ uint64_t d_siphash_k0;
__constant__ uint64_t d_siphash_k1;
__constant__ uint64_t d_siphash_k2;
__constant__ uint64_t d_siphash_k3;


////////////////////////////////////////////////////////////////////////////////
// Helper: saturating 2-bit atomic increment
// Packs 16 two-bit counters per uint32_t word.
// Saturates at value 3 (= "≥ 3").
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void atomicInc2Sat(uint32_t* base, uint32_t nodeId)
{
    uint32_t* word    = base + (nodeId >> 4);
    uint32_t  shift   = (nodeId & 15u) << 1;
    uint32_t  satMask = 3u << shift;

    uint32_t old = *word;
    uint32_t desired;
    do
    {
        uint32_t val = (old >> shift) & 3u;
        if (val == 3u) { return; }              // already saturated
        desired = (old & ~satMask) | ((val + 1u) << shift);
    } while (atomicCAS(word, old, desired) != old
             && (old = *word, true));           // retry with fresh read
}


////////////////////////////////////////////////////////////////////////////////
// Helper: read 2-bit counter
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
uint32_t read2(uint32_t const* base, uint32_t nodeId)
{
    return (base[nodeId >> 4] >> ((nodeId & 15u) << 1)) & 3u;
}


////////////////////////////////////////////////////////////////////////////////
// Kernel: count node degrees into 2-bit packed counters
//
// @param edgeBitmap  [in]  alive-edge bitmap
// @param nodeDegree  [out] 2-bit degree counters (must be pre-zeroed)
// @param uorv        0 = count U-nodes, 1 = count V-nodes
////////////////////////////////////////////////////////////////////////////////
__global__
void cuckatoo32CountDegreeKernel(
    uint32_t const* __restrict__ edgeBitmap,
    uint32_t*       __restrict__ nodeDegree,
    uint32_t                     uorv)
{
    uint64_t const stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    uint64_t       e      = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    for (; e < algo::cuckatoo::NUM_EDGES; e += stride)
    {
        // Check if edge is alive
        if (!((edgeBitmap[e >> 5] >> (e & 31u)) & 1u)) { continue; }

        // Compute endpoint node
        uint32_t const node = (uorv == 0u)
            ? sipNodeU(d_siphash_k0, d_siphash_k1, d_siphash_k2, d_siphash_k3, e)
            : sipNodeV(d_siphash_k0, d_siphash_k1, d_siphash_k2, d_siphash_k3, e);

        atomicInc2Sat(nodeDegree, node);
    }
}


////////////////////////////////////////////////////////////////////////////////
// Kernel: trim edges whose endpoint has degree < 2
//
// @param edgeBitmap  [in/out] alive-edge bitmap (bits cleared here)
// @param nodeDegree  [in]     2-bit degree counters from count kernel
// @param uorv        0 = trim by U-degree, 1 = trim by V-degree
////////////////////////////////////////////////////////////////////////////////
__global__
void cuckatoo32TrimEdgesKernel(
    uint32_t*       __restrict__ edgeBitmap,
    uint32_t const* __restrict__ nodeDegree,
    uint32_t                     uorv)
{
    uint64_t const stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    uint64_t       e      = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    for (; e < algo::cuckatoo::NUM_EDGES; e += stride)
    {
        if (!((edgeBitmap[e >> 5] >> (e & 31u)) & 1u)) { continue; }

        uint32_t const node = (uorv == 0u)
            ? sipNodeU(d_siphash_k0, d_siphash_k1, d_siphash_k2, d_siphash_k3, e)
            : sipNodeV(d_siphash_k0, d_siphash_k1, d_siphash_k2, d_siphash_k3, e);

        if (read2(nodeDegree, node) < 2u)
        {
            atomicAnd(&edgeBitmap[e >> 5], ~(1u << (e & 31u)));
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
// Kernel: compact surviving edges into a dense array
////////////////////////////////////////////////////////////////////////////////
__global__
void cuckatoo32CompactKernel(
    uint32_t const* __restrict__ edgeBitmap,
    uint32_t*       __restrict__ edgeList,
    uint32_t*       __restrict__ edgeCount,
    uint32_t                     maxEdges)
{
    uint64_t const stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    uint64_t       e      = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    for (; e < algo::cuckatoo::NUM_EDGES; e += stride)
    {
        if ((edgeBitmap[e >> 5] >> (e & 31u)) & 1u)
        {
            uint32_t const idx = atomicAdd(edgeCount, 1u);
            if (idx < maxEdges)
            {
                edgeList[idx] = static_cast<uint32_t>(e);
            }
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
// Host function implementations
////////////////////////////////////////////////////////////////////////////////

bool cuckatoo32AllocMemory(resolver::nvidia::cuckatoo32::KernelParameters& params)
{
    using P = resolver::nvidia::cuckatoo32::KernelParameters;

    uint64_t const bitmapBytes      = P::EDGE_BITMAP_WORDS * sizeof(uint32_t);   // 512 MB
    uint64_t const nodeDegreeBytes  = P::NODE_DEGREE_WORDS * sizeof(uint32_t);   // 1 GB
    uint64_t const edgeListBytes    = P::MAX_EDGES_COMPACT  * sizeof(uint32_t);  //  4 MB

    if (nullptr != params.devEdgeBitmap) { return true; }  // already allocated

    CUDA_ER(cudaMalloc(&params.devEdgeBitmap, bitmapBytes));
    CUDA_ER(cudaMalloc(&params.devNodeDegree, nodeDegreeBytes));
    CUDA_ER(cudaMalloc(&params.devEdgeList,   edgeListBytes));
    CUDA_ER(cudaMalloc(&params.devEdgeCount,  sizeof(uint32_t)));

    return true;
}


bool cuckatoo32FreeMemory(resolver::nvidia::cuckatoo32::KernelParameters& params)
{
    if (nullptr != params.devEdgeBitmap)
    {
        cudaFree(params.devEdgeBitmap);
        params.devEdgeBitmap = nullptr;
    }
    if (nullptr != params.devNodeDegree)
    {
        cudaFree(params.devNodeDegree);
        params.devNodeDegree = nullptr;
    }
    if (nullptr != params.devEdgeList)
    {
        cudaFree(params.devEdgeList);
        params.devEdgeList = nullptr;
    }
    if (nullptr != params.devEdgeCount)
    {
        cudaFree(params.devEdgeCount);
        params.devEdgeCount = nullptr;
    }
    return true;
}


bool cuckatoo32UpdateConstants(resolver::nvidia::cuckatoo32::KernelParameters& params)
{
    CUDA_ER(cudaMemcpyToSymbol(d_siphash_k0, &params.k0, sizeof(uint64_t)));
    CUDA_ER(cudaMemcpyToSymbol(d_siphash_k1, &params.k1, sizeof(uint64_t)));
    CUDA_ER(cudaMemcpyToSymbol(d_siphash_k2, &params.k2, sizeof(uint64_t)));
    CUDA_ER(cudaMemcpyToSymbol(d_siphash_k3, &params.k3, sizeof(uint64_t)));
    return true;
}


bool cuckatoo32Trim(
    cudaStream_t                                    stream,
    resolver::nvidia::cuckatoo32::KernelParameters& params)
{
    using P = resolver::nvidia::cuckatoo32::KernelParameters;

    constexpr uint32_t BLOCKS  { algo::cuckatoo::DEFAULT_BLOCKS  };
    constexpr uint32_t THREADS { algo::cuckatoo::DEFAULT_THREADS };

    // Seed: set all edges alive (bitmap = all-ones)
    CUDA_ER(cudaMemsetAsync(
        params.devEdgeBitmap, 0xFF,
        P::EDGE_BITMAP_WORDS * sizeof(uint32_t),
        stream));

    // 2 × TRIM_ROUNDS half-rounds (alternating U and V sides)
    for (uint32_t round = 0u; round < algo::cuckatoo::TRIM_ROUNDS * 2u; ++round)
    {
        uint32_t const uorv = round & 1u;   // 0 for U, 1 for V

        // Zero degree counters
        CUDA_ER(cudaMemsetAsync(
            params.devNodeDegree, 0,
            P::NODE_DEGREE_WORDS * sizeof(uint32_t),
            stream));

        // Count node degrees
        cuckatoo32CountDegreeKernel<<<BLOCKS, THREADS, 0, stream>>>(
            params.devEdgeBitmap,
            params.devNodeDegree,
            uorv);
        CUDA_ER(cudaGetLastError());

        // Trim dead edges
        cuckatoo32TrimEdgesKernel<<<BLOCKS, THREADS, 0, stream>>>(
            params.devEdgeBitmap,
            params.devNodeDegree,
            uorv);
        CUDA_ER(cudaGetLastError());
    }

    // Wait for all trimming to finish
    CUDA_ER(cudaStreamSynchronize(stream));

    return true;
}


////////////////////////////////////////////////////////////////////////////////
// CPU cycle detection helpers
////////////////////////////////////////////////////////////////////////////////

/// CPU-side SipHash-2-4 (same algorithm as GPU)
static uint64_t cpuRotl64(uint64_t x, int n) { return (x << n) | (x >> (64 - n)); }

#define CPU_SIPROUND(v0,v1,v2,v3)                      \
    do {                                               \
        (v0)+=(v1); (v1)=cpuRotl64((v1),13)^(v0);    \
        (v0)=cpuRotl64((v0),32);                       \
        (v2)+=(v3); (v3)=cpuRotl64((v3),16)^(v2);    \
        (v0)+=(v3); (v3)=cpuRotl64((v3),21)^(v0);    \
        (v2)+=(v1); (v1)=cpuRotl64((v1),17)^(v2);    \
        (v2)=cpuRotl64((v2),32);                       \
    } while (0)

static uint64_t cpuSipHash24(
    uint64_t k0, uint64_t k1, uint64_t k2, uint64_t k3, uint64_t nonce)
{
    uint64_t v0 = k0, v1 = k1, v2 = k2, v3 = k3 ^ nonce;
    CPU_SIPROUND(v0,v1,v2,v3); CPU_SIPROUND(v0,v1,v2,v3);
    v0 ^= nonce; v2 ^= 0xffull;
    CPU_SIPROUND(v0,v1,v2,v3); CPU_SIPROUND(v0,v1,v2,v3);
    CPU_SIPROUND(v0,v1,v2,v3); CPU_SIPROUND(v0,v1,v2,v3);
    return v0 ^ v1 ^ v2 ^ v3;
}
#undef CPU_SIPROUND

static uint32_t cpuSipNodeU(uint64_t k0, uint64_t k1, uint64_t k2, uint64_t k3, uint64_t e)
{
    return static_cast<uint32_t>(cpuSipHash24(k0, k1, k2, k3, e * 2ULL));
}
static uint32_t cpuSipNodeV(uint64_t k0, uint64_t k1, uint64_t k2, uint64_t k3, uint64_t e)
{
    return static_cast<uint32_t>(cpuSipHash24(k0, k1, k2, k3, e * 2ULL + 1ULL));
}


////////////////////////////////////////////////////////////////////////////////
/// DFS-based 42-cycle finder on CPU.
/// @param edges  compact list of surviving edge indices
/// @param count  number of surviving edges
/// @param k0..k3 SipHash keys to recompute (u,v) for each edge
/// @param proof  output: sorted 42 edge indices forming the cycle
/// @returns true if a cycle was found
////////////////////////////////////////////////////////////////////////////////
static bool findCycle42(
    uint32_t const* edges,
    uint32_t        count,
    uint64_t        k0, uint64_t k1, uint64_t k2, uint64_t k3,
    uint32_t        proof[algo::cuckatoo::PROOF_SIZE])
{
    if (count < algo::cuckatoo::PROOF_SIZE) { return false; }
    if (count > resolver::nvidia::cuckatoo32::KernelParameters::MAX_EDGES_COMPACT)
    {
        logWarn() << "cuckatoo32FindCycle: too many surviving edges (" << count
                  << "), skipping cycle search";
        return false;
    }

    // Build adjacency list: node → list of (neighbourNode, edgeIndex)
    using AdjEntry = std::pair<uint32_t, uint32_t>; // (otherNode, edgeIdx)
    std::unordered_map<uint32_t, std::vector<AdjEntry>> adj;
    adj.reserve(count * 2u);

    for (uint32_t i = 0u; i < count; ++i)
    {
        uint32_t const e = edges[i];
        uint32_t const u = cpuSipNodeU(k0, k1, k2, k3, e);
        uint32_t const v = cpuSipNodeV(k0, k1, k2, k3, e);
        adj[u].emplace_back(v | 0x80000000u, i);   // encode V-node with high bit set
        adj[v | 0x80000000u].emplace_back(u, i);
    }

    // DFS: find a simple cycle of length exactly PROOF_SIZE
    constexpr uint32_t TARGET_DEPTH{ algo::cuckatoo::PROOF_SIZE };

    std::vector<uint32_t> path;
    path.reserve(TARGET_DEPTH);
    std::vector<bool> usedEdge(count, false);
    bool found = false;

    for (auto const& [startNode, _startAdj] : adj)
    {
        if (found) { break; }

        path.clear();
        std::fill(usedEdge.begin(), usedEdge.end(), false);

        // Iterative DFS via explicit stack to avoid stack overflow
        struct Frame { uint32_t node; uint32_t adjIdx; };
        std::vector<Frame> stack;
        stack.reserve(TARGET_DEPTH + 1u);
        stack.push_back({ startNode, 0u });

        while (!stack.empty() && !found)
        {
            Frame& top = stack.back();
            auto it    = adj.find(top.node);

            if (it == adj.end() || top.adjIdx >= it->second.size())
            {
                // Backtrack
                if (!path.empty())
                {
                    usedEdge[path.back()] = false;
                    path.pop_back();
                }
                stack.pop_back();
                continue;
            }

            auto const& [nextNode, edgeIdx] = it->second[top.adjIdx];
            ++top.adjIdx;

            if (usedEdge[edgeIdx]) { continue; }

            if (path.size() + 1u == TARGET_DEPTH)
            {
                // Check if this closes a 42-cycle back to startNode
                if (nextNode == startNode)
                {
                    path.push_back(edgeIdx);
                    // Build sorted proof from edge indices in path
                    for (uint32_t j = 0u; j < TARGET_DEPTH; ++j)
                    {
                        proof[j] = edges[path[j]];
                    }
                    std::sort(proof, proof + TARGET_DEPTH);
                    found = true;
                }
                continue;
            }

            // Descend
            usedEdge[edgeIdx] = true;
            path.push_back(edgeIdx);
            stack.push_back({ nextNode, 0u });
        }
    }

    return found;
}


bool cuckatoo32FindCycle(
    resolver::nvidia::cuckatoo32::KernelParameters& params,
    bool*                                           outFound,
    uint32_t                                        outProof[algo::cuckatoo::PROOF_SIZE])
{
    // Reset compact edge counter on GPU
    uint32_t const zero = 0u;
    CUDA_ER(cudaMemcpy(params.devEdgeCount, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Launch compact kernel to gather surviving edge indices
    constexpr uint32_t BLOCKS  { algo::cuckatoo::DEFAULT_BLOCKS  };
    constexpr uint32_t THREADS { algo::cuckatoo::DEFAULT_THREADS };

    cuckatoo32CompactKernel<<<BLOCKS, THREADS>>>(
        params.devEdgeBitmap,
        params.devEdgeList,
        params.devEdgeCount,
        resolver::nvidia::cuckatoo32::KernelParameters::MAX_EDGES_COMPACT);
    CUDA_ER(cudaGetLastError());
    CUDA_ER(cudaDeviceSynchronize());

    // Copy count and edge list to host
    uint32_t edgeCount = 0u;
    CUDA_ER(cudaMemcpy(&edgeCount, params.devEdgeCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (edgeCount == 0u || edgeCount > resolver::nvidia::cuckatoo32::KernelParameters::MAX_EDGES_COMPACT)
    {
        return true;  // No cycle possible, but not an error
    }

    std::vector<uint32_t> edgesHost(edgeCount);
    CUDA_ER(cudaMemcpy(
        edgesHost.data(), params.devEdgeList,
        edgeCount * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Search for 42-cycle on CPU
    *outFound = findCycle42(
        edgesHost.data(), edgeCount,
        params.k0, params.k1, params.k2, params.k3,
        outProof);

    return true;
}
