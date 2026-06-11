// Host harness that runs the kHeavyHash OpenCL kernels on whatever OpenCL device the
// ICD exposes (POCL/CPU in the cross-build & dev harness, a real GPU on the rig) and
// asserts each stage is BIT-IDENTICAL to the CPU reference's known-answer vectors.
//
// The program is assembled and compiled through common::KernelGeneratorOpenCL -- the
// same path the production resolver uses -- instead of a hand-rolled
// clCreateProgramWithSource. The shipped .cl is loaded at runtime (self-contained, no
// inner #include), so there is no second copy of the kernel to drift. CL errors are
// decoded with opencl_error.hpp (openclShowError).

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "kheavyhash_test_vectors.hpp"
#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/kheavyhash/matrix.hpp>
#include <algo/kheavyhash/result.hpp>
#include <common/error/opencl_error.hpp>
#include <common/kernel_generator/opencl.hpp>


using kheavyhash::Hash256;
using kheavyhash::HASH_SIZE;
using kheavyhash::Matrix;
using kheavyhash::MATRIX_DIM;


static std::vector<uint16_t> flatten(Matrix const& matrix)
{
    std::vector<uint16_t> flat{};
    flat.reserve(MATRIX_DIM * MATRIX_DIM);
    for (size_t r{ 0 }; r < MATRIX_DIM; ++r)
    {
        for (size_t c{ 0 }; c < MATRIX_DIM; ++c)
        {
            flat.push_back(matrix[r][c]);
        }
    }
    return flat;
}


// Resolve the shipped .cl: env override (fast iteration on the rig) first, then the
// compile-time in-tree source path (POCL/dev), then the deployed path next to the
// binary. First that opens wins, so the same unit_test runs in the dev harness and on
// a real GPU. Returns "" if none open.
static std::string resolveKernelPath()
{
    char const* const candidates[3]{ std::getenv("KH_CL_PATH"), KH_CL_PATH, "kernel/kheavyhash/kheavyhash.cl" };
    for (char const* const cand : candidates)
    {
        if (nullptr != cand)
        {
            std::ifstream const file{ cand };
            if (file.good())
            {
                return cand;
            }
        }
    }
    return "";
}


// One built program, shared by every test; each test pulls its kernel by name.
class OpenClKat : public ::testing::Test
{
  protected:
    static cl::Device                    device;
    static cl::Context                   context;
    static cl::CommandQueue              queue;
    static common::KernelGeneratorOpenCL generator;

    static void SetUpTestSuite()
    {
        try
        {
            std::vector<cl::Platform> platforms{};
            cl::Platform::get(&platforms);
            ASSERT_FALSE(platforms.empty()) << "no OpenCL platform";

            std::vector<cl::Device> devices{};
            platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
            ASSERT_FALSE(devices.empty()) << "no OpenCL device";

            device = devices[0];
            context = cl::Context(device);
            queue = cl::CommandQueue(context, device);

            std::string const clPath{ resolveKernelPath() };
            ASSERT_FALSE(clPath.empty()) << "cannot open kernel source (tried env KH_CL_PATH, " << KH_CL_PATH
                                         << ", kernel/kheavyhash/kheavyhash.cl)";

            generator.clear();
            generator.setKernelName("search");
            generator.addDefine("MAX_RESULT", algo::kheavyhash::MAX_RESULT);
            ASSERT_TRUE(generator.appendFile(clPath)) << "cannot append kernel source: " << clPath;
            ASSERT_TRUE(generator.build(&device, &context)) << "kernel build failed (see build log above)";
        }
        catch (cl::Error const& clErr)
        {
            FAIL() << openclShowError(clErr.err()) << " - " << clErr.what();
        }
    }

    cl::Kernel kernel(char const* name)
    {
        return cl::Kernel(generator.clProgram, name);
    }

    void run(cl::Kernel& k, size_t const global)
    {
        queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(global), cl::NullRange);
        queue.finish();
    }
};

cl::Device                    OpenClKat::device{};
cl::Context                   OpenClKat::context{};
cl::CommandQueue              OpenClKat::queue{};
common::KernelGeneratorOpenCL OpenClKat::generator{};


static Hash256 toHash(std::array<uint8_t, HASH_SIZE> const& a)
{
    Hash256 h{};
    for (size_t i{ 0 }; i < HASH_SIZE; ++i)
    {
        h[i] = a[i];
    }
    return h;
}


TEST_F(OpenClKat, powHashMatchesReference)
{
    try
    {
        std::array<uint8_t, HASH_SIZE> pre{};
        for (size_t i{ 0 }; i < HASH_SIZE; ++i)
        {
            pre[i] = kheavyhash::kat::POW_KAT_PRE[i];
        }
        std::array<uint8_t, HASH_SIZE> out{};

        cl::Buffer preBuf{ context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pre.size(), pre.data() };
        cl::Buffer outBuf{ context, CL_MEM_WRITE_ONLY, out.size() };
        cl::Kernel k{ kernel("test_pow_hash") };
        cl_ulong   ts{ kheavyhash::kat::POW_KAT_TIMESTAMP };
        cl_ulong   nonce{ kheavyhash::kat::POW_KAT_NONCE };
        k.setArg(0u, preBuf);
        k.setArg(1u, ts);
        k.setArg(2u, nonce);
        k.setArg(3u, outBuf);
        run(k, 1);
        queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, out.size(), out.data());

        EXPECT_EQ(out, kheavyhash::kat::POW_KAT_EXPECTED);
    }
    catch (cl::Error const& clErr)
    {
        FAIL() << openclShowError(clErr.err()) << " - " << clErr.what();
    }
}


TEST_F(OpenClKat, kHeavyHashMatchesReference)
{
    try
    {
        std::array<uint8_t, HASH_SIZE> out{};
        cl::Buffer                     inBuf{ context,
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          kheavyhash::kat::HEAVY_INPUT.size(),
                          const_cast<uint8_t*>(kheavyhash::kat::HEAVY_INPUT.data()) };
        cl::Buffer                     outBuf{ context, CL_MEM_WRITE_ONLY, out.size() };
        cl::Kernel                     k{ kernel("test_kheavy") };
        k.setArg(0u, inBuf);
        k.setArg(1u, outBuf);
        run(k, 1);
        queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, out.size(), out.data());

        EXPECT_EQ(out, kheavyhash::kat::KHEAVY_EXPECTED);
    }
    catch (cl::Error const& clErr)
    {
        FAIL() << openclShowError(clErr.err()) << " - " << clErr.what();
    }
}


TEST_F(OpenClKat, heavyHashMatchesReference)
{
    try
    {
        std::vector<uint16_t> const    flat{ flatten(kheavyhash::kat::HEAVY_TEST_MATRIX) };
        std::array<uint8_t, HASH_SIZE> out{};

        cl::Buffer matBuf{ context,
                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           flat.size() * sizeof(uint16_t),
                           const_cast<uint16_t*>(flat.data()) };
        cl::Buffer h1Buf{ context,
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          kheavyhash::kat::HEAVY_INPUT.size(),
                          const_cast<uint8_t*>(kheavyhash::kat::HEAVY_INPUT.data()) };
        cl::Buffer outBuf{ context, CL_MEM_WRITE_ONLY, out.size() };
        cl::Kernel k{ kernel("test_heavy_hash") };
        k.setArg(0u, matBuf);
        k.setArg(1u, h1Buf);
        k.setArg(2u, outBuf);
        run(k, 1);
        queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, out.size(), out.data());

        EXPECT_EQ(out, kheavyhash::kat::HEAVY_EXPECTED);
    }
    catch (cl::Error const& clErr)
    {
        FAIL() << openclShowError(clErr.err()) << " - " << clErr.what();
    }
}


// Full per-nonce mining kernel: given the job's matrix/header/target, the work item
// whose nonce reproduces FP_FINAL must report a hit at exactly FP_NONCE, and must NOT
// report one against a target one unit below FP_FINAL.
class SearchKernel : public OpenClKat, public ::testing::WithParamInterface<char const*>
{
  protected:
    // Mirrors the kernel's Result struct (and algo::ethash::Result layout).
    struct alignas(8) Result
    {
        uint8_t  found{ 0u };
        uint32_t count{ 0u };
        uint64_t nonces[algo::kheavyhash::MAX_RESULT]{ 0ull, 0ull, 0ull, 0ull };
    };

    cl_ulong
    runSearch(std::array<uint8_t, HASH_SIZE> const& target, cl_ulong nonceStart, size_t globalSize, cl_uint& foundCount)
    {
        Matrix const                matrix{ kheavyhash::generateMatrix(toHash(kheavyhash::kat::FP_PRE)) };
        std::vector<uint16_t> const flat{ flatten(matrix) };

        Result     result{};
        cl::Buffer matBuf{ context,
                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           flat.size() * sizeof(uint16_t),
                           const_cast<uint16_t*>(flat.data()) };
        cl::Buffer preBuf{ context,
                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           kheavyhash::kat::FP_PRE.size(),
                           const_cast<uint8_t*>(kheavyhash::kat::FP_PRE.data()) };
        cl::Buffer tgtBuf{ context,
                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           target.size(),
                           const_cast<uint8_t*>(target.data()) };
        cl::Buffer resBuf{ context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(Result), &result };

        cl::Kernel k{ kernel(GetParam()) };
        cl_ulong   ts{ kheavyhash::kat::FP_TIMESTAMP };
        k.setArg(0u, matBuf);
        k.setArg(1u, preBuf);
        k.setArg(2u, tgtBuf);
        k.setArg(3u, ts);
        k.setArg(4u, nonceStart);
        k.setArg(5u, resBuf);
        run(k, globalSize);
        queue.enqueueReadBuffer(resBuf, CL_TRUE, 0, sizeof(Result), &result);

        foundCount = result.count;
        return result.nonces[0];
    }
};


// FP_TARGET_PASS == FP_FINAL and FP_TARGET_FAIL == FP_FINAL - 1 (LSB decremented). A
// single work-item at exactly FP_NONCE that passes the first and fails the second pins
// the kernel's end-to-end pow output to FP_FINAL bit-for-bit, and proves the
// hit-reporting (atomic_inc + foundNonce = nonceStart + gid) path.
TEST_P(SearchKernel, reportsHitAtWinningNonce)
{
    try
    {
        cl_uint        count{ 0 };
        cl_ulong const found{ runSearch(kheavyhash::kat::FP_TARGET_PASS, kheavyhash::kat::FP_NONCE, 1, count) };

        EXPECT_EQ(1u, count);
        EXPECT_EQ(kheavyhash::kat::FP_NONCE, found);
    }
    catch (cl::Error const& clErr)
    {
        FAIL() << openclShowError(clErr.err()) << " - " << clErr.what();
    }
}


TEST_P(SearchKernel, noHitWhenPowExceedsTarget)
{
    try
    {
        cl_uint count{ 0 };
        runSearch(kheavyhash::kat::FP_TARGET_FAIL, kheavyhash::kat::FP_NONCE, 1, count);

        EXPECT_EQ(0u, count);
    }
    catch (cl::Error const& clErr)
    {
        FAIL() << openclShowError(clErr.err()) << " - " << clErr.what();
    }
}


// The shipped `search` kernel is pinned bit-identical against the CPU reference: the
// winning-nonce / boundary vectors must hold end-to-end. A drift of a single bit in its
// LDS staging, udot4 matmul, or unrolled keccak fails here. The optimisation variants
// that led to it are exercised by the AMD benchmark.
INSTANTIATE_TEST_SUITE_P(
    Search,
    SearchKernel,
    ::testing::Values("search"),
    [](::testing::TestParamInfo<char const*> const& info)
    {
        return std::string{ info.param };
    });
