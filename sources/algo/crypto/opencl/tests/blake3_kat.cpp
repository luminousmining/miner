// Known-answer test for the shared BLAKE3 OpenCL primitive
// (sources/algo/crypto/opencl/blake3.cl). Runs blake3_hash_chunk on whatever OpenCL
// device the ICD exposes (POCL/CPU in the cross-build & dev harness, a real GPU on the
// rig) and asserts it is BIT-IDENTICAL to the vendored reference
// (crypto/reference/blake3, the official BLAKE3 C implementation) for several
// single-chunk input sizes, including a 64-byte output that exercises the XOF upper
// words.
//
// The device program is assembled through common::KernelGeneratorOpenCL, chaining the
// real shipped primitive then the test wrapper via appendFile (no #include resolution,
// no second copy of the kernel). CL errors are decoded with opencl_error.hpp.

#include <cstdint>
#include <cstring>
#include <vector>

#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include "blake3.h"  // vendored reference, via blake3_ref

#include <common/error/opencl_error.hpp>
#include <common/kernel_generator/opencl.hpp>


namespace
{
    // Prefer the kernel deployed next to the binary (shipped exe / CWD with a kernel/
    // tree); fall back to the in-tree source path baked at compile time. This reproduces
    // the old readKernel dual-path lookup so the KAT runs where no kernel/ tree sits next
    // to unit_test (POCL/dev). appendFile checks the stream BEFORE appending, so a failed
    // first try leaves the assembled source untouched.
    bool appendWithFallback(common::KernelGeneratorOpenCL& gen, char const* deployed, char const* inTree)
    {
        if (true == gen.appendFile(deployed))
        {
            return true;
        }
        return gen.appendFile(inTree);
    }


    class Blake3SharedKat : public ::testing::Test
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

                device  = devices[0];
                context = cl::Context(device);
                queue   = cl::CommandQueue(context, device);

                generator.clear();
                generator.setKernelName("blake3_kat");
                ASSERT_TRUE(appendWithFallback(generator, "kernel/crypto/blake3.cl", BLAKE3_CRYPTO_CL_PATH))
                    << "cannot open crypto blake3.cl (tried kernel/crypto/blake3.cl and " << BLAKE3_CRYPTO_CL_PATH
                    << ")";
                ASSERT_TRUE(appendWithFallback(generator, "kernel/crypto/tests/blake3_kat.cl", BLAKE3_KAT_CL_PATH))
                    << "cannot open test wrapper blake3_kat.cl (tried kernel/crypto/tests/blake3_kat.cl and "
                    << BLAKE3_KAT_CL_PATH << ")";
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

        // Hashes `len` bytes of a deterministic pattern on the device and compares to
        // the vendored reference for the requested digest length.
        void runOne(uint32_t const len, uint32_t const outlen)
        {
            try
            {
                std::vector<uint8_t> in(len);
                for (uint32_t i{ 0 }; i < len; ++i)
                {
                    in[i] = static_cast<uint8_t>((i * 7u + 13u) & 0xFFu);
                }

                uint8_t       expected[64]{};
                blake3_hasher hasher;
                blake3_hasher_init(&hasher);
                blake3_hasher_update(&hasher, in.data(), len);
                blake3_hasher_finalize(&hasher, expected, outlen);

                std::size_t const inBytes{ (0u == len) ? std::size_t{ 1 } : std::size_t{ len } };
                cl::Buffer        inBuf{ context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inBytes,
                                  (0u == len) ? expected : in.data() };
                cl::Buffer        outBuf{ context, CL_MEM_WRITE_ONLY, std::size_t{ outlen } };

                cl::Kernel k{ kernel("blake3_kat") };
                k.setArg(0u, inBuf);
                k.setArg(1u, len);
                k.setArg(2u, outlen);
                k.setArg(3u, outBuf);

                queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(1), cl::NullRange);
                queue.finish();

                uint8_t got[64]{};
                queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, outlen, got);

                EXPECT_EQ(0, std::memcmp(got, expected, outlen)) << "mismatch at len=" << len << " outlen=" << outlen;
            }
            catch (cl::Error const& clErr)
            {
                FAIL() << openclShowError(clErr.err()) << " - " << clErr.what();
            }
        }
    };

    cl::Device                    Blake3SharedKat::device{};
    cl::Context                   Blake3SharedKat::context{};
    cl::CommandQueue              Blake3SharedKat::queue{};
    common::KernelGeneratorOpenCL Blake3SharedKat::generator{};
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
