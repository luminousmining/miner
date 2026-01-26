#include <common/error/cuda_error.hpp>
#include <common/error/nvrtc_error.hpp>
#include <common/kernel_generator/cuda.hpp>
#include <common/log/log.hpp>
#include <common/cast.hpp>
#include <common/chrono.hpp>


bool common::KernelGeneratorCuda::build(
    uint32_t const deviceId,
    uint32_t const major,
    uint32_t const minor)
{
    ////////////////////////////////////////////////////////////////////////////
    std::string fullSource;
    common::Chrono chrono;

    ////////////////////////////////////////////////////////////////////////////
    declareDefine("__KERNEL_COMPILED");
    declareDefine("__LIB_CUDA");

    ////////////////////////////////////////////////////////////////////////////
    chrono.start();

    ////////////////////////////////////////////////////////////////////////////
    for (auto const& kv : defines)
    {
        fullSource += "#define " + kv.first + " " + kv.second + "\n";
    }
    fullSource += "\n";
    fullSource += sourceCode;

    ////////////////////////////////////////////////////////////////////////////
    writeKernelInFile(fullSource, ".cu");

    ////////////////////////////////////////////////////////////////////////////
    NVRTC_ER(
        nvrtcCreateProgram(
            &cuProgram,
            fullSource.c_str(),
            kernelName.c_str(),
            0,
            nullptr,
            nullptr
    ));
    NVRTC_ER(nvrtcAddNameExpression(cuProgram, kernelName.c_str()));

    // Compilation
    std::string flagGpuArchitecture
    {
        "--gpu-architecture=compute_"
        + std::to_string(major)
        + std::to_string(minor)
    };
    const char* flags[]
    {
        flagGpuArchitecture.c_str(),
        "-rdc=true",
        "-use_fast_math",
        "-std=c++17"
    };

    ////////////////////////////////////////////////////////////////////////
    int const nFlagsCount { sizeof(flags) / sizeof(flags[0]) };
    NVRTC_BUILD(nvrtcCompileProgram(cuProgram, nFlagsCount, flags));

    ////////////////////////////////////////////////////////////////////////
    size_t ptxSize{ 0ull };
    NVRTC_ER(nvrtcGetPTXSize(cuProgram, &ptxSize));
    char* ptx{ NEW_ARRAY(char, ptxSize) };
    NVRTC_ER(nvrtcGetPTX(cuProgram, ptx));

    ////////////////////////////////////////////////////////////////////////
    CUmodule moduleData;
    const char* mangledName{ nullptr };
    CU_ER(cuModuleLoadDataEx(&moduleData, ptx, 0, nullptr, nullptr));
    NVRTC_ER(nvrtcGetLoweredName(cuProgram, kernelName.c_str(), &mangledName));
    CU_ER(cuModuleGetFunction(&cuFunction, moduleData, mangledName));

    ////////////////////////////////////////////////////////////////////////
    chrono.stop();
    logInfo()
        << "GPU["<< deviceId << "]"
        << " built kernel " << kernelName
        << " in " << chrono.elapsed(common::CHRONO_UNIT::MS) << "ms";

    ////////////////////////////////////////////////////////////////////////
    sourceCode.clear();

    ////////////////////////////////////////////////////////////////////////
    NVRTC_ER(nvrtcDestroyProgram(&cuProgram));
    SAFE_DELETE_ARRAY(ptx);

    ////////////////////////////////////////////////////////////////////////
    built = true;
    return true;
}


bool common::KernelGeneratorCuda::occupancy(
    CUdevice* const cuDevice,
    uint32_t const threads,
    uint32_t const blocks)
{
    CU_ER(
        cuFuncGetAttribute(
            &maxThreads,
            CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            cuFunction));
    CU_ER(
        cuDeviceGetAttribute(
            &maxBlocks,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            *cuDevice));

    if (   threads != castU32(maxThreads)
        || blocks != castU32(maxBlocks))
    {
        return true;
    }

    return false;
}
