#pragma once

#include <map>
#include <string>
#include <list>


namespace common
{
    struct KernelGenerator
    {
    public:
        int32_t maxThreads{ 1 };
        int32_t maxBlocks{ 1 };

        virtual void clear();
        void setKernelName(std::string const& kernelFunctionName);
        void addInclude(std::string const& pathInclude);
        void declareDefine(std::string const& name);
        void appendLine(std::string const& line);
        bool appendFile(std::string const& pathFileName);
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

    protected:
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