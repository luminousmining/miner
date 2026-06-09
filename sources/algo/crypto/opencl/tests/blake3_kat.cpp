// Known-answer test for the shared BLAKE3 OpenCL primitive
// (sources/algo/crypto/opencl/blake3.cl). Runs blake3_hash_chunk on whatever OpenCL
// device the ICD exposes (POCL/CPU in CI, a real GPU otherwise) and asserts it is
// BIT-IDENTICAL to the vendored reference (crypto/reference/blake3, the official
// BLAKE3 C implementation) for several single-chunk input sizes, including a 64-byte
// output that exercises the XOF upper words.
//
// The device program is the real shipped primitive (BLAKE3_CRYPTO_CL_PATH)
// concatenated with the test wrapper (BLAKE3_KAT_CL_PATH), so there is no second copy
// of the kernel to drift.

#define CL_TARGET_OPENCL_VERSION 300

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <CL/cl.h>
#include <gtest/gtest.h>

#include "blake3.h"  // vendored reference, via blake3_ref


namespace
{
    void clCheck(cl_int const err, char const* what)
    {
        ASSERT_EQ(CL_SUCCESS, err) << what << " failed: " << err;
    }


    // The shipped exe carries the kernels next to it (kernel/...), but the in-tree
    // source paths (BLAKE3_*_CL_PATH) only exist in the build tree. Prefer the deployed
    // copy so the test runs from an installed/cross-built artifact; fall back to source.
    std::string readKernel(char const* deployed, char const* fallback)
    {
        std::ifstream in{ deployed };
        if (false == in.good())
        {
            in.clear();
            in.open(fallback);
        }
        EXPECT_TRUE(in.good()) << "cannot open kernel source (tried " << deployed << " and " << fallback << ")";
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }


    class Blake3SharedKat : public ::testing::Test
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

            std::string const src{ readKernel("kernel/crypto/blake3.cl", BLAKE3_CRYPTO_CL_PATH) + "\n"
                                   + readKernel("kernel/crypto/tests/blake3_kat.cl", BLAKE3_KAT_CL_PATH) };
            char const*  srcPtr{ src.c_str() };
            size_t const srcLen{ src.size() };

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

        // Hashes `len` bytes of a deterministic pattern on the device and compares to
        // the vendored reference for the requested digest length.
        void runOne(uint32_t const len, uint32_t const outlen)
        {
            std::vector<uint8_t> in(len);
            for (uint32_t i{ 0 }; i < len; ++i)
            {
                in[i] = static_cast<uint8_t>((i * 7u + 13u) & 0xFFu);
            }

            uint8_t expected[64]{};
            blake3_hasher hasher;
            blake3_hasher_init(&hasher);
            blake3_hasher_update(&hasher, in.data(), len);
            blake3_hasher_finalize(&hasher, expected, outlen);

            cl_int err{ CL_SUCCESS };
            cl_mem inBuf{ clCreateBuffer(context,
                                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         (0u == len) ? 1u : len,
                                         (0u == len) ? expected : in.data(),
                                         &err) };
            clCheck(err, "clCreateBuffer in");
            cl_mem outBuf{ clCreateBuffer(context, CL_MEM_WRITE_ONLY, outlen, nullptr, &err) };
            clCheck(err, "clCreateBuffer out");

            cl_kernel k{ clCreateKernel(program, "blake3_kat", &err) };
            clCheck(err, "clCreateKernel");
            clCheck(clSetKernelArg(k, 0, sizeof(cl_mem), &inBuf), "arg0");
            clCheck(clSetKernelArg(k, 1, sizeof(cl_uint), &len), "arg1");
            clCheck(clSetKernelArg(k, 2, sizeof(cl_uint), &outlen), "arg2");
            clCheck(clSetKernelArg(k, 3, sizeof(cl_mem), &outBuf), "arg3");

            size_t global{ 1 };
            clCheck(clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "ndr");
            clCheck(clFinish(queue), "finish");

            uint8_t got[64]{};
            clCheck(clEnqueueReadBuffer(queue, outBuf, CL_TRUE, 0, outlen, got, 0, nullptr, nullptr), "read");

            EXPECT_EQ(0, std::memcmp(got, expected, outlen)) << "mismatch at len=" << len << " outlen=" << outlen;

            clReleaseKernel(k);
            clReleaseMemObject(inBuf);
            clReleaseMemObject(outBuf);
        }
    };

    cl_context       Blake3SharedKat::context{ nullptr };
    cl_command_queue Blake3SharedKat::queue{ nullptr };
    cl_program       Blake3SharedKat::program{ nullptr };
    cl_device_id     Blake3SharedKat::device{ nullptr };
}


TEST_F(Blake3SharedKat, EmptyInput)
{
    runOne(0u, 32u);
}


TEST_F(Blake3SharedKat, OneBlock64)
{
    runOne(64u, 32u);
}


TEST_F(Blake3SharedKat, NinetySix)
{
    runOne(96u, 32u);
}


TEST_F(Blake3SharedKat, XofSeed180)
{
    runOne(180u, 64u);
}
