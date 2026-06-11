// Host-compiled known-answer test of the CUDA device hash functions.
//
// kheavyhash_device.cuh is written so that, when compiled WITHOUT nvcc
// (__CUDACC__ undefined), its __device__ functions become plain host functions.
// That lets us verify the CUDA source logic against the same vectors as the CPU
// reference and the OpenCL kernel, on a machine with no NVIDIA GPU. The GPU
// launch glue (search.cuh / memory.cuh) still needs real hardware to run.

#include <array>
#include <cstdint>
#include <vector>

#include "kheavyhash_test_vectors.hpp"
#include <gtest/gtest.h>

#include <algo/kheavyhash/cuda/kheavyhash_device.cuh>
#include <algo/kheavyhash/matrix.hpp>

using kheavyhash::HASH_SIZE;
using kheavyhash::MATRIX_DIM;


static std::array<uint8_t, HASH_SIZE> toArr(uint8_t const* p)
{
    std::array<uint8_t, HASH_SIZE> a{};
    for (size_t i{ 0 }; i < HASH_SIZE; ++i)
    {
        a[i] = p[i];
    }
    return a;
}


static std::vector<uint16_t> flatten(kheavyhash::Matrix const& m)
{
    std::vector<uint16_t> f;
    f.reserve(MATRIX_DIM * MATRIX_DIM);
    for (size_t r{ 0 }; r < MATRIX_DIM; ++r)
    {
        for (size_t c{ 0 }; c < MATRIX_DIM; ++c)
        {
            f.push_back(m[r][c]);
        }
    }
    return f;
}


TEST(CudaDeviceKat, powHashMatchesReference)
{
    uint8_t out[HASH_SIZE];
    kheavyhash_cuda::powHash(
        kheavyhash::kat::POW_KAT_PRE,
        kheavyhash::kat::POW_KAT_TIMESTAMP,
        kheavyhash::kat::POW_KAT_NONCE,
        out);
    EXPECT_EQ(toArr(out), kheavyhash::kat::POW_KAT_EXPECTED);
}


TEST(CudaDeviceKat, kHeavyHashMatchesReference)
{
    uint8_t out[HASH_SIZE];
    kheavyhash_cuda::kHeavyHash(kheavyhash::kat::HEAVY_INPUT.data(), out);
    EXPECT_EQ(toArr(out), kheavyhash::kat::KHEAVY_EXPECTED);
}


TEST(CudaDeviceKat, heavyHashMatchesReference)
{
    std::vector<uint16_t> flat;
    flat.reserve(MATRIX_DIM * MATRIX_DIM);
    for (auto const& row : kheavyhash::kat::HEAVY_TEST_MATRIX)
    {
        for (uint16_t const v : row)
        {
            flat.push_back(v);
        }
    }
    uint8_t out[HASH_SIZE];
    kheavyhash_cuda::heavyHash(flat.data(), kheavyhash::kat::HEAVY_INPUT.data(), out);
    EXPECT_EQ(toArr(out), kheavyhash::kat::HEAVY_EXPECTED);
}


TEST(CudaDeviceKat, fullPipelineMatchesReference)
{
    // The matrix is generated host-side (CPU reference); the CUDA device funcs do
    // powHash -> heavyHash, mirroring the on-GPU per-nonce path.
    kheavyhash::Matrix const    matrix{ kheavyhash::generateMatrix(kheavyhash::kat::FP_PRE) };
    std::vector<uint16_t> const flat{ flatten(matrix) };

    uint8_t h1[HASH_SIZE];
    kheavyhash_cuda::powHash(
        kheavyhash::kat::FP_PRE.data(),
        kheavyhash::kat::FP_TIMESTAMP,
        kheavyhash::kat::FP_NONCE,
        h1);
    EXPECT_EQ(toArr(h1), kheavyhash::kat::FP_HASH1);

    uint8_t pow[HASH_SIZE];
    kheavyhash_cuda::heavyHash(flat.data(), h1, pow);
    EXPECT_EQ(toArr(pow), kheavyhash::kat::FP_FINAL);
}


TEST(CudaDeviceKat, meetsTargetCompare)
{
    EXPECT_TRUE(kheavyhash_cuda::meetsTarget(kheavyhash::kat::FP_FINAL.data(), kheavyhash::kat::FP_TARGET_PASS.data()));
    EXPECT_FALSE(
        kheavyhash_cuda::meetsTarget(kheavyhash::kat::FP_FINAL.data(), kheavyhash::kat::FP_TARGET_FAIL.data()));
}
