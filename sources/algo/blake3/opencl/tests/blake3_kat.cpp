// Host harness that runs the Blake3 (Alephium) OpenCL kernels on whatever OpenCL
// device the ICD exposes (POCL/CPU in the cross-build & dev harness, a real GPU on
// the rig) and asserts the kernel is BIT-IDENTICAL to the host reference
// (sources/algo/blake3/blake3_pow.cpp).
//
// The program is assembled and compiled the same way the production resolver builds
// it: through common::KernelGeneratorOpenCL, chaining the shared crypto primitive then
// the Alephium mining kernel via appendFile (no #include resolution, no second copy of
// the kernel). CL errors are decoded with opencl_error.hpp (openclShowError).

#include <cstdint>
#include <cstring>
#include <vector>

#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/blake3/blake3_pow.hpp>
#include <algo/blake3/result.hpp>
#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <common/error/opencl_error.hpp>
#include <common/kernel_generator/opencl.hpp>


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


    struct alignas(8) Result
    {
        uint8_t  found{ 0 };
        uint32_t count{ 0 };
        uint64_t nonces[4]{ 0, 0, 0, 0 };
    };


    class Blake3Cl : public ::testing::Test
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
                generator.setKernelName("search");
                generator.addDefine("MAX_RESULT", algo::blake3::MAX_RESULT);
                ASSERT_TRUE(generator.appendFile("kernel/common/rotate_byte.cl"))
                    << "cannot open kernel/common/rotate_byte.cl";
                ASSERT_TRUE(generator.appendFile("kernel/crypto/blake3.cl")) << "cannot open kernel/crypto/blake3.cl";
                ASSERT_TRUE(generator.appendFile("kernel/blake3/blake3.cl")) << "cannot open kernel/blake3/blake3.cl";
                ASSERT_TRUE(generator.build(&device, &context)) << "kernel build failed (see build log above)";
            }
            catch (cl::Error const& clErr)
            {
                FAIL() << openclShowError(clErr.err()) << " - " << clErr.what();
            }
        }

        cl::Kernel makeKernel(char const* name)
        {
            return cl::Kernel(generator.clProgram, name);
        }
    };

    cl::Device                    Blake3Cl::device{};
    cl::Context                   Blake3Cl::context{};
    cl::CommandQueue              Blake3Cl::queue{};
    common::KernelGeneratorOpenCL Blake3Cl::generator{};
}


TEST_F(Blake3Cl, TestHashMatchesHostReference)
{
    try
    {
        algo::hash3072 const header{ algo::toHash<algo::hash3072>(HEADER_HEX, algo::HASH_SHIFT::LEFT) };
        algo::hash256        expected{};
        algo::blake3::hashRef(header, NONCE, expected);

        uint32_t   out[8]{};
        cl::Buffer headerBuffer{ context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(header),
                                 const_cast<uint32_t*>(header.word32) };
        cl::Buffer outBuffer{ context, CL_MEM_WRITE_ONLY, sizeof(out) };

        cl::Kernel kernel{ makeKernel("test_hash") };
        cl_ulong   nonce{ NONCE };
        kernel.setArg(0u, headerBuffer);
        kernel.setArg(1u, nonce);
        kernel.setArg(2u, outBuffer);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange);
        queue.finish();
        queue.enqueueReadBuffer(outBuffer, CL_TRUE, 0, sizeof(out), out);

        EXPECT_EQ(0, std::memcmp(out, expected.word32, sizeof(out)));
    }
    catch (cl::Error const& clErr)
    {
        FAIL() << openclShowError(clErr.err()) << " - " << clErr.what();
    }
}


TEST_F(Blake3Cl, SearchReportsHitAtWinningNonce)
{
    try
    {
        algo::hash3072 const header{ algo::toHash<algo::hash3072>(HEADER_HEX, algo::HASH_SHIFT::LEFT) };
        algo::hash256        digest{};
        algo::blake3::hashRef(header, NONCE, digest);

        uint32_t const bigIndex{ algo::blake3::chainIndex(digest) };
        uint32_t const fromGroup{ bigIndex / 4u };
        uint32_t const toGroup{ bigIndex % 4u };

        Result     result{};
        cl::Buffer headerBuffer{ context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(header),
                                 const_cast<uint32_t*>(header.word32) };
        cl::Buffer targetBuffer{ context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(digest.ubytes),
                                 digest.ubytes };
        cl::Buffer resultBuffer{ context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(Result), &result };

        cl::Kernel kernel{ makeKernel("search") };
        cl_ulong   startNonce{ NONCE };
        kernel.setArg(0u, headerBuffer);
        kernel.setArg(1u, targetBuffer);
        kernel.setArg(2u, resultBuffer);
        kernel.setArg(3u, startNonce);
        kernel.setArg(4u, fromGroup);
        kernel.setArg(5u, toGroup);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange);
        queue.finish();
        queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, sizeof(Result), &result);

        EXPECT_EQ(1u, result.count);
        EXPECT_EQ(NONCE, result.nonces[0]);
    }
    catch (cl::Error const& clErr)
    {
        FAIL() << openclShowError(clErr.err()) << " - " << clErr.what();
    }
}
