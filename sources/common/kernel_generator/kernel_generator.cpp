#if defined(__linux__)
    #include <experimental/filesystem>
    namespace __fs = std::experimental::filesystem;
#else
    #include <filesystem>
    namespace __fs = std::filesystem;
#endif
#include <fstream>


#include <common/kernel_generator/kernel_generator.hpp>
#include <common/log/log.hpp>


void common::KernelGenerator::clear()
{
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
