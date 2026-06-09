// Host harness that runs the Blake3 (Alephium) OpenCL kernels on whatever OpenCL
// device the ICD exposes (POCL/CPU in CI, a real GPU otherwise) and asserts the
// kernel is BIT-IDENTICAL to the host reference (sources/algo/blake3/blake3_pow.cpp).
//
// The shipped .cl is loaded at runtime (path baked in as BLAKE3_CL_PATH), so there
// is no second copy of the kernel to drift.

#define CL_TARGET_OPENCL_VERSION 300

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include <CL/cl.h>
#include <gtest/gtest.h>

#include <algo/blake3/blake3_pow.hpp>
#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>


namespace
{
    constexpr char const* HEADER_HEX{
        "000700000000000022d30e3358af8cd1a732e46d47254bb81ffa43cf402dfe001cf5000000000001bd38686272dfd4b55c3559391a"
        "dfab12413c197fa92729465aea000000000001ea959d9fbdd9abeda14a65c0040bb7e626d7c13a6e51979a434f000000000002776a"
        "5133a5941c8bf35e38043a1f4c5700c0844cb835c414c4300000000000017266826c86467877ca053f0776c56a9d340f634237a91e"
        "3865410000000000009dc9c87ce69910b950d0967c9a37930fcfd34f510fc0a014861200000000000157c2d1db2a2af17d3e7560bf"
        "c78b270d2b7e65a7fcff312b3b8310249f0949d3463117e80375771047ab3309102f365ddc609b36eaae1363dfceb25811b3902af4"
        "dd41490d75b7f79a0478f7f721681c59178a2564b84561559a0000018ea81c4bfa1b029ed6" };
    constexpr uint64_t NONCE{ 0x914544566c9a0a4dull };

    void clCheck(cl_int const err, char const* what)
    {
        ASSERT_EQ(CL_SUCCESS, err) << what << " failed: " << err;
    }


    struct alignas(8) Result
    {
        uint8_t  found{ 0 };
        uint32_t count{ 0 };
        uint64_t nonces[4]{ 0, 0, 0, 0 };
    };


    class Blake3Cl : public ::testing::Test
    {
      protected:
        static cl_context       context;
        static cl_command_queue queue;
        static cl_program       program;
        static cl_device_id     device;

        static void SetUpTestSuite()
        {
            cl_platform_id platform{};
            cl_uint        numPlatforms{ 0 };
            clCheck(clGetPlatformIDs(1, &platform, &numPlatforms), "clGetPlatformIDs");
            ASSERT_GT(numPlatforms, 0u) << "no OpenCL platform";

            cl_uint numDevices{ 0 };
            clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &numDevices), "clGetDeviceIDs");
            ASSERT_GT(numDevices, 0u) << "no OpenCL device";

            cl_int err{ CL_SUCCESS };
            context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
            clCheck(err, "clCreateContext");
            queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
            clCheck(err, "clCreateCommandQueue");

            // Prefer the kernel deployed next to the binary (works for the shipped
            // exe); fall back to the in-tree source path baked at build time (CI).
            std::ifstream in{ "kernel/blake3/blake3.cl" };
            if (false == in.good())
            {
                in.clear();
                in.open(BLAKE3_CL_PATH);
            }
            ASSERT_TRUE(in.good()) << "cannot open kernel source (tried kernel/blake3/blake3.cl and " << BLAKE3_CL_PATH
                                   << ")";
            std::stringstream ss;
            ss << in.rdbuf();
            std::string const src{ ss.str() };
            char const*       srcPtr{ src.c_str() };
            size_t const      srcLen{ src.size() };

            program = clCreateProgramWithSource(context, 1, &srcPtr, &srcLen, &err);
            clCheck(err, "clCreateProgramWithSource");
            err = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
            if (CL_SUCCESS != err)
            {
                size_t logSize{ 0 };
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
                std::string log(logSize, '\0');
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
                FAIL() << "clBuildProgram failed:\n" << log;
            }
        }

        static void TearDownTestSuite()
        {
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }

        cl_mem makeBuf(cl_mem_flags const flags, size_t const bytes, void const* host)
        {
            cl_int err{ CL_SUCCESS };
            cl_mem buf{ clCreateBuffer(context, flags, bytes, const_cast<void*>(host), &err) };
            clCheck(err, "clCreateBuffer");
            return buf;
        }

        cl_kernel kernel(char const* name)
        {
            cl_int    err{ CL_SUCCESS };
            cl_kernel k{ clCreateKernel(program, name, &err) };
            clCheck(err, "clCreateKernel");
            return k;
        }
    };

    cl_context       Blake3Cl::context{ nullptr };
    cl_command_queue Blake3Cl::queue{ nullptr };
    cl_program       Blake3Cl::program{ nullptr };
    cl_device_id     Blake3Cl::device{ nullptr };
}


TEST_F(Blake3Cl, TestHashMatchesHostReference)
{
    algo::hash3072 const header{ algo::toHash<algo::hash3072>(HEADER_HEX, algo::HASH_SHIFT::LEFT) };
    algo::hash256        expected{};
    algo::blake3::hashRef(header, NONCE, expected);

    uint32_t out[8]{};
    cl_mem   hdrBuf{ makeBuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(header), const_cast<uint32_t*>(header.word32)) };
    cl_mem   outBuf{ makeBuf(CL_MEM_WRITE_ONLY, sizeof(out), nullptr) };
    cl_kernel k{ kernel("test_hash") };
    cl_ulong  nonce{ NONCE };
    clCheck(clSetKernelArg(k, 0, sizeof(cl_mem), &hdrBuf), "arg0");
    clCheck(clSetKernelArg(k, 1, sizeof(cl_ulong), &nonce), "arg1");
    clCheck(clSetKernelArg(k, 2, sizeof(cl_mem), &outBuf), "arg2");
    size_t global{ 1 };
    clCheck(clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "ndr");
    clCheck(clFinish(queue), "finish");
    clCheck(clEnqueueReadBuffer(queue, outBuf, CL_TRUE, 0, sizeof(out), out, 0, nullptr, nullptr), "read");

    EXPECT_EQ(0, std::memcmp(out, expected.word32, sizeof(out)));

    clReleaseKernel(k);
    clReleaseMemObject(hdrBuf);
    clReleaseMemObject(outBuf);
}


TEST_F(Blake3Cl, SearchReportsHitAtWinningNonce)
{
    algo::hash3072 const header{ algo::toHash<algo::hash3072>(HEADER_HEX, algo::HASH_SHIFT::LEFT) };
    algo::hash256        digest{};
    algo::blake3::hashRef(header, NONCE, digest);

    uint32_t const bigIndex{ algo::blake3::chainIndex(digest) };
    uint32_t const fromGroup{ bigIndex / 4u };
    uint32_t const toGroup{ bigIndex % 4u };

    Result result{};
    cl_mem hdrBuf{ makeBuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(header), const_cast<uint32_t*>(header.word32)) };
    cl_mem tgtBuf{ makeBuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32, digest.ubytes) };
    cl_mem resBuf{ makeBuf(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(Result), &result) };

    cl_kernel k{ kernel("search") };
    cl_ulong  startNonce{ NONCE };
    clCheck(clSetKernelArg(k, 0, sizeof(cl_mem), &hdrBuf), "arg0");
    clCheck(clSetKernelArg(k, 1, sizeof(cl_mem), &tgtBuf), "arg1");
    clCheck(clSetKernelArg(k, 2, sizeof(cl_ulong), &startNonce), "arg2");
    clCheck(clSetKernelArg(k, 3, sizeof(cl_uint), &fromGroup), "arg3");
    clCheck(clSetKernelArg(k, 4, sizeof(cl_uint), &toGroup), "arg4");
    clCheck(clSetKernelArg(k, 5, sizeof(cl_mem), &resBuf), "arg5");
    size_t global{ 1 };
    clCheck(clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "ndr");
    clCheck(clFinish(queue), "finish");
    clCheck(clEnqueueReadBuffer(queue, resBuf, CL_TRUE, 0, sizeof(Result), &result, 0, nullptr, nullptr), "read");

    EXPECT_EQ(1u, result.count);
    EXPECT_EQ(NONCE, result.nonces[0]);

    clReleaseKernel(k);
    clReleaseMemObject(hdrBuf);
    clReleaseMemObject(tgtBuf);
    clReleaseMemObject(resBuf);
}
