#if defined(__linux__)
    #include <experimental/filesystem>
    namespace __fs = std::experimental::filesystem;
#else
    #include <filesystem>
    namespace __fs = std::filesystem;
#endif
#include <fstream>

#include <CL/opencl.hpp>

#include <common/chrono.hpp>
#include <common/kernel_generator.hpp>
#include <common/log/log.hpp>
#include <common/error/opencl_error.hpp>
#include <common/error/cuda_error.hpp>
#include <common/error/nvrtc_error.hpp>


void common::KernelGenerator::clear()
{
#if defined(AMD_ENABLE)
    // OpenCL
    clKernel = nullptr;
    clProgram = nullptr;
#endif

    // Common
    sourceCode.clear();
    kernelName.clear();
    compileFlags.clear();
    defines.clear();
    includes.clear();
}


void common::KernelGenerator::setKernelName(
    std::string const& kernelFunctionName)
{
    kernelName.assign(kernelFunctionName);
}


void common::KernelGenerator::addInclude(
    std::string const& pathInclude)
{
    includes.push_back(pathInclude);
}


void common::KernelGenerator::declareDefine(
    std::string const& name)
{
    defines[name].clear();
}


void common::KernelGenerator::appendLine(
    std::string const& line)
{
    sourceCode += line;
    sourceCode += "\n";
}


bool common::KernelGenerator::appendFile(
    std::string const& pathFileName)
{
    std::ifstream ifs{pathFileName};
    if (false == ifs.good())
    {
        logErr() << "Can not append file " << pathFileName;
        return false;
    }

    sourceCode += "///////////////////////////////////////////////////////////////////////////////";
    sourceCode += "\n";
    sourceCode += "// File: " + pathFileName;
    sourceCode += "\n";
    sourceCode += "///////////////////////////////////////////////////////////////////////////////";
    sourceCode += "\n";
    std::string line;
    while (std::getline(ifs, line))
    {
        appendLine(line);
    }

    sourceCode += "///////////////////////////////////////////////////////////////////////////////";
    sourceCode += "\n";
    sourceCode += "\n";

    return true;
}


#if defined(AMD_ENABLE)
bool common::KernelGenerator::buildOpenCL(
    cl::Device* const clDevice,
    cl::Context* const clContext)
{
    try
    {
        ////////////////////////////////////////////////////////////////////////
        std::string fullSource;
        common::Chrono chrono;

        ////////////////////////////////////////////////////////////////////////
        chrono.start();

        ////////////////////////////////////////////////////////////////////////
        if (false == defines.empty())
        {
            fullSource += "///////////////////////////////////////////////////////////////////////////////";
            fullSource += "\n";
            for (auto const& kv : defines)
            {
                fullSource += "#define " + kv.first + " " + kv.second + "\n";
            }
            fullSource += "///////////////////////////////////////////////////////////////////////////////";
            fullSource += "\n";
            fullSource += "\n";
        }

        ////////////////////////////////////////////////////////////////////////
        if (false == includes.empty())
        {
            fullSource += "///////////////////////////////////////////////////////////////////////////////";
            fullSource += "\n";
            for (auto const& pathInclude : includes)
            {
                fullSource += "#include \"";
                fullSource += pathInclude;
                fullSource += "\"";
                fullSource += "\n";
            }
            fullSource += "///////////////////////////////////////////////////////////////////////////////";
            fullSource += "\n";
            fullSource += "\n";
        }

        ////////////////////////////////////////////////////////////////////////
        fullSource += sourceCode;

        ////////////////////////////////////////////////////////////////////////
        writeKernelInFile(fullSource, ".cl");

        ////////////////////////////////////////////////////////////////////////
        clProgram = cl::Program(*clContext, fullSource);

        ////////////////////////////////////////////////////////////////////////
        compileFlags += " -cl-fast-relaxed-math";
        compileFlags += " -cl-no-signed-zeros";
        compileFlags += " -cl-mad-enable";
        compileFlags += " -cl-std=CL2.0";
#if defined(__DEBUG)
        compileFlags += " -O0";
#else
        compileFlags += " -O3";
#endif
        compileFlags += " -I ./";

        ////////////////////////////////////////////////////////////////////////
        clProgram.build(*clDevice, compileFlags.c_str());
        clKernel = cl::Kernel(clProgram, kernelName.c_str());

        ////////////////////////////////////////////////////////////////////////
        chrono.stop();
        logInfo()
            << "Build kernel " << kernelName
            << " in " << chrono.elapsed(common::CHRONO_UNIT::MS) << "ms";

        ////////////////////////////////////////////////////////////////////////
        sourceCode.clear();
    }
    catch(cl::BuildError const& clErr)
    {
        auto const clBuildStatus{ clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*clDevice) };
        logErr()
            << "(" << __FUNCTION__ << ")"
            << "{" << openclShowError(clErr.err()) << "}"
            << " -> " << kernelName
            << "\n"
            << clBuildStatus;
        return false;
    }
    catch (cl::Error const& clErr)
    {
        OPENCL_EXCEPTION_ERROR_SHOW(__FUNCTION__, clErr);
        return false;
    }

    ////////////////////////////////////////////////////////////////////////
    built = true;
    return built;
}
#endif


#if defined(CUDA_ENABLE)
bool common::KernelGenerator::buildCuda(
    CUdevice* const cuDevice,
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
    char* ptx { new char[ptxSize] };
    NVRTC_ER(nvrtcGetPTX(cuProgram, ptx));

    ////////////////////////////////////////////////////////////////////////
    CUmodule moduleData;
    const char* mangledName{ nullptr };
    CU_ER(cuModuleLoadDataEx(&moduleData, ptx, 0, nullptr, nullptr));
    NVRTC_ER(nvrtcGetLoweredName(cuProgram, kernelName.c_str(), &mangledName));
    CU_ER(cuModuleGetFunction(&cuFunction, moduleData, mangledName));

    ////////////////////////////////////////////////////////////////////////
    CU_ER(cuFuncGetAttribute(&maxThreads,
                             CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                             cuFunction));
    CU_ER(cuDeviceGetAttribute(&maxBlocks,
                               CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                               *cuDevice));

    ////////////////////////////////////////////////////////////////////////
    chrono.stop();
    logInfo()
        << "Build kernel " << kernelName
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
#endif


bool common::KernelGenerator::isBuilt() const
{
    return built;
}


void common::KernelGenerator::writeKernelInFile(
    std::string const& source,
    std::string const& extension)
{
    __fs::path pathKernel { "kernel" };
    pathKernel /= "kernel_" + kernelName + extension;
    __fs::create_directories(pathKernel.parent_path());

    logDebug() << "Write kernel " << pathKernel;

    std::ofstream ofs { pathKernel };
    ofs << source.c_str();
    ofs.close();
}