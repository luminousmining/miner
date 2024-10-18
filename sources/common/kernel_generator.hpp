#pragma once

#include <map>
#include <string>
#include <list>

#if defined(CUDA_ENABLE)
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <nvrtc.h>
#endif
#if defined(AMD_ENABLE)
    #include <CL/opencl.hpp>
#endif


namespace common
{
    struct KernelGenerator
    {
    public:
#if defined(CUDA_ENABLE)
        ////////////////////////////////////////////////////////////////////
        CUfunction cuFunction{ nullptr };
#endif
#if defined(AMD_ENABLE)
        ////////////////////////////////////////////////////////////////////
        cl::Kernel clKernel{};
#endif

        void clear();
        void setKernelName(std::string const& kernelFunctionName);
        void addInclude(std::string const& pathInclude);
        void declareDefine(std::string const& name);
        void appendLine(std::string const& line);
        bool appendFile(std::string const& pathFileName);
#if defined(AMD_ENABLE)
        bool buildOpenCL(cl::Device* const clDevice,
                        cl::Context* const clContext);
#endif
#if defined(CUDA_ENABLE)
        bool buildCuda(uint32_t const major,
                       uint32_t miner);
#endif
        bool isBuilt() const;

        template<typename T>
        inline
        void addDefine(std::string const& name,
                       T const& value)
        {
            static_assert(
                   std::is_same<T, uint8_t>()
                || std::is_same<T, uint16_t>()
                || std::is_same<T, uint32_t>()
                || std::is_same<T, uint64_t>()
                || std::is_same<T, int8_t>()
                || std::is_same<T, int16_t>()
                || std::is_same<T, int32_t>()
                || std::is_same<T, int64_t>());
            defines[name] = std::to_string(value);
        }

    private:
#if defined(CUDA_ENABLE)
        ////////////////////////////////////////////////////////////////////
        nvrtcProgram cuProgram{};
#endif
        ////////////////////////////////////////////////////////////////////
#if defined(AMD_ENABLE)
        cl::Program  clProgram{};
#endif
        ////////////////////////////////////////////////////////////////////
        bool built { false };
        std::string  kernelName{};
        std::string  compileFlags{};
        std::string  sourceCode{};
        std::list<std::string> includes{};
        std::map<std::string, std::string> defines{};

        void writeKernelInFile(std::string const& source,
                               std::string const& extension);
    };
}