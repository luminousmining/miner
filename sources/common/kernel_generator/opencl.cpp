#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <common/error/opencl_error.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <common/log/log.hpp>
#include <common/chrono.hpp>


void common::KernelGeneratorOpenCL::clear()
{
    clKernel = nullptr;
    clProgram = nullptr;

    common::KernelGenerator::clear();
}


bool common::KernelGeneratorOpenCL::build(
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
