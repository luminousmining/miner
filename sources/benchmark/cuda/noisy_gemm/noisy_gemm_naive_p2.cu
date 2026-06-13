///////////////////////////////////////////////////////////////////////////////
// Pearl NoisyGEMM — naive kernel p2
// Tile shape: tm=32, tn=32, noise rank r=64
// Larger tiles to measure register pressure impact vs p1.
///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

#include <benchmark/cuda/noisy_gemm/noisy_gemm_device.cuh>
#include <common/error/cuda_error.hpp>


///////////////////////////////////////////////////////////////////////////////
constexpr uint32_t TILE_HEIGHT{ 32u }; // tm — tile rows
constexpr uint32_t TILE_WIDTH { 32u }; // tn — tile columns
constexpr uint32_t NOISE_RANK { 64u }; // r  — low-rank noise dimension


///////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
uint32_t rotl32_p2(uint32_t const x, uint32_t const n)
{
    return __funnelshift_l(x, x, n);
}


///////////////////////////////////////////////////////////////////////////////
// NoisyGEMM naive kernel p2.
// Each thread processes one tile (tileRow, tileCol) independently.
//
// Glossary (Pearl whitepaper §4):
//   tileRow    — first row index of this tile in A' (multiple of TILE_HEIGHT)
//   tileCol    — first column index of this tile in B' (multiple of TILE_WIDTH)
//   rows       — actual tile height (< TILE_HEIGHT at bottom boundary)
//   cols       — actual tile width  (< TILE_WIDTH  at right boundary)
//   step       — index over k/NOISE_RANK accumulation steps (l in the spec)
//   kOffset    — starting k-index for the current step (s = step * NOISE_RANK)
//   accumState — M[16], 512-bit rolling hash updated after each complete tile step
//   xorReduce  — XOR of all INT32 values in Cblk, mixed into accumState
__global__
void kernel_noisy_gemm_naive_p2(
    int8_t   const* __restrict__ dA,
    int8_t   const* __restrict__ dB,
    int32_t*        __restrict__ dC,
    uint32_t const               m,
    uint32_t const               n,
    uint32_t const               k,
    uint8_t  const* __restrict__ sA,
    uint64_t const* __restrict__ threshold,
    uint32_t*                    dWinningCount,
    algo::noisy_gemm::WinningTileGpu* dWinning,
    uint32_t const               maxWinning)
{
    uint32_t const tilesPerRow{ (n + TILE_WIDTH - 1u) / TILE_WIDTH };
    uint32_t const tileIndex  { blockIdx.x * blockDim.x + threadIdx.x };
    uint32_t const tileRow    { (tileIndex / tilesPerRow) * TILE_HEIGHT };
    uint32_t const tileCol    { (tileIndex % tilesPerRow) * TILE_WIDTH  };

    if (tileRow >= m || tileCol >= n)
    {
        return;
    }

    uint32_t const rows{ min(TILE_HEIGHT, m - tileRow) };
    uint32_t const cols{ min(TILE_WIDTH,  n - tileCol) };

    int32_t  Cblk[TILE_HEIGHT][TILE_WIDTH]{};
    uint32_t accumState[16]{};

    uint32_t const numSteps{ k / NOISE_RANK };

    for (uint32_t step{ 0u }; step < numSteps; ++step)
    {
        uint32_t const kOffset{ step * NOISE_RANK };

        for (uint32_t ii{ 0u }; ii < rows; ++ii)
        {
            for (uint32_t jj{ 0u }; jj < cols; ++jj)
            {
                int32_t acc{ 0 };
                for (uint32_t kk{ 0u }; kk < NOISE_RANK; ++kk)
                {
                    int32_t const a{ static_cast<int32_t>(dA[(tileRow + ii) * k + kOffset + kk]) };
                    int32_t const b{ static_cast<int32_t>(dB[(kOffset + kk) * n + tileCol + jj]) };
                    acc += a * b;
                }
                Cblk[ii][jj] += acc;
            }
        }

        if (rows == TILE_HEIGHT && cols == TILE_WIDTH)
        {
            int32_t xorReduce{ 0 };
            for (uint32_t ii{ 0u }; ii < TILE_HEIGHT; ++ii)
            {
                for (uint32_t jj{ 0u }; jj < TILE_WIDTH; ++jj)
                {
                    xorReduce ^= Cblk[ii][jj];
                }
            }
            accumState[step % 16u] = rotl32_p2(accumState[step % 16u], 13u)
                                   ^ static_cast<uint32_t>(xorReduce);
        }
    }

    for (uint32_t ii{ 0u }; ii < rows; ++ii)
    {
        for (uint32_t jj{ 0u }; jj < cols; ++jj)
        {
            dC[(tileRow + ii) * n + tileCol + jj] = Cblk[ii][jj];
        }
    }

    if (rows == TILE_HEIGHT && cols == TILE_WIDTH)
    {
        uint8_t hash[32]{};
        noisy_gemm::device::blake3HashM(accumState, sA, hash);

        if (true == noisy_gemm::device::checkPow(hash, threshold))
        {
            uint32_t const idx{ atomicAdd(dWinningCount, 1u) };
            if (idx < maxWinning)
            {
                algo::noisy_gemm::WinningTileGpu& out{ dWinning[idx] };
                out.tile_i = tileRow;
                out.tile_j = tileCol;
                for (uint32_t i{ 0u }; i < 16u; ++i) { out.M[i]      = accumState[i]; }
                for (uint32_t i{ 0u }; i < 32u; ++i) { out.M_hash[i] = hash[i];       }
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////
__host__
bool pearl_naive_p2(
    cudaStream_t                        stream,
    int8_t const*                       dA,
    int8_t const*                       dB,
    int32_t*                            dC,
    uint32_t                            m,
    uint32_t                            n,
    uint32_t                            k,
    uint8_t const*                      sA,
    uint64_t const*                     threshold,
    uint32_t*                           dWinningCount,
    algo::noisy_gemm::WinningTileGpu*   dWinning,
    uint32_t                            maxWinning,
    uint32_t                            blocks,
    uint32_t                            threads)
{
    kernel_noisy_gemm_naive_p2<<<blocks, threads, 0, stream>>>(
        dA, dB, dC, m, n, k, sA, threshold, dWinningCount, dWinning, maxWinning);
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());
    return true;
}
