#pragma once

#include <map>
#include <string>
#include <list>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <CL/opencl.hpp>


namespace common
{
    struct KernelGenerator
    {
    public:
        ////////////////////////////////////////////////////////////////////
        CUfunction cuFunction{ nullptr };
        ////////////////////////////////////////////////////////////////////
        cl::Kernel clKernel{};

        void clear();
        void setKernelName(std::string const& kernelFunctionName);
        void addInclude(std::string const& pathInclude);
        void declareDefine(std::string const& name);
        void appendLine(std::string const& line);
        bool appendFile(std::string const& pathFileName);
        bool buildOpenCL(cl::Device* const clDevice,
                        cl::Context* const clContext);
        bool buildCuda(uint32_t const major,
                       uint32_t miner);
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
        ////////////////////////////////////////////////////////////////////
        nvrtcProgram cuProgram{};
        ////////////////////////////////////////////////////////////////////
        cl::Program  clProgram{};
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