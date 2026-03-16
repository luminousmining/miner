#include <fstream>
#include <sstream>

#include <boost/json.hpp>

#include <benchmark/config.hpp>
#include <common/log/log.hpp>


benchmark::KernelParams parseKernelParams(
    boost::json::object const& obj,
    benchmark::KernelParams const& defaults)
{
    benchmark::KernelParams p{ defaults };
    if (obj.contains("loop"))    { p.loop    = static_cast<uint32_t>(obj.at("loop").to_number<uint64_t>()); }
    if (obj.contains("threads")) { p.threads = static_cast<uint32_t>(obj.at("threads").to_number<uint64_t>()); }
    if (obj.contains("blocks"))  { p.blocks  = static_cast<uint32_t>(obj.at("blocks").to_number<uint64_t>()); }
    return p;
}

benchmark::AlgoConfig parseAlgoConfig(
    boost::json::object const& obj,
    benchmark::AlgoConfig const& defaultAlgo)
{
    benchmark::AlgoConfig algo{ defaultAlgo };

    if (obj.contains("enabled")) { algo.enabled          = obj.at("enabled").as_bool(); }
    if (obj.contains("loop"))    { algo.defaults.loop    = static_cast<uint32_t>(obj.at("loop").to_number<uint64_t>()); }
    if (obj.contains("threads")) { algo.defaults.threads = static_cast<uint32_t>(obj.at("threads").to_number<uint64_t>()); }
    if (obj.contains("blocks"))  { algo.defaults.blocks  = static_cast<uint32_t>(obj.at("blocks").to_number<uint64_t>()); }

    if (obj.contains("kernels"))
    {
        algo.kernels.clear();
        for (auto const& item : obj.at("kernels").as_array())
        {
            if (item.is_string())
            {
                std::string const name{ item.as_string() };
                algo.kernels.emplace_back(name, algo.defaults);
            }
            else if (item.is_object())
            {
                auto const& kernelObj{ item.as_object() };
                std::string const name{ kernelObj.at("name").as_string() };
                algo.kernels.emplace_back(name, parseKernelParams(kernelObj, algo.defaults));
            }
        }
    }

    return algo;
}

benchmark::VendorConfig parseVendorConfig(
    boost::json::object const& obj,
    benchmark::VendorConfig const& defaultVendor)
{
    benchmark::VendorConfig vendor{ defaultVendor };

    if (obj.contains("enabled"))      { vendor.enabled     = obj.at("enabled").as_bool(); }
    if (obj.contains("device_index")) { vendor.deviceIndex = static_cast<uint32_t>(obj.at("device_index").to_number<uint64_t>()); }

    if (obj.contains("algorithms"))
    {
        for (auto const& algoItem : obj.at("algorithms").as_object())
        {
            std::string const name{ algoItem.key() };
            benchmark::AlgoConfig defaultAlgo{};
            auto it{ vendor.algorithms.find(name) };
            if (it != vendor.algorithms.end())
            {
                defaultAlgo = it->second;
            }
            vendor.algorithms[name] = parseAlgoConfig(algoItem.value().as_object(), defaultAlgo);
        }
    }

    return vendor;
}


// cppcheck-suppress unusedFunction
bool benchmark::AlgoConfig::isKernelEnabled(std::string const& name) const
{
    if (kernels.empty())
    {
        return true;
    }
    for (auto const& kernel : kernels)
    {
        if (kernel.first == name)
        {
            return true;
        }
    }
    return false;
}


// cppcheck-suppress unusedFunction
benchmark::KernelParams benchmark::AlgoConfig::resolveKernel(std::string const& name) const
{
    for (auto const& kernel : kernels)
    {
        if (kernel.first == name)
        {
            return kernel.second;
        }
    }
    return defaults;
}


// cppcheck-suppress unusedFunction
bool benchmark::VendorConfig::isAlgoEnabled(std::string const& name) const
{
    if (false == enabled)
    {
        return false;
    }
    auto const it{ algorithms.find(name) };
    if (it == algorithms.end())
    {
        return true;
    }
    return it->second.enabled;
}


// cppcheck-suppress unusedFunction
benchmark::AlgoConfig const& benchmark::VendorConfig::getAlgo(std::string const& name) const
{
    static benchmark::AlgoConfig const fallback{};
    auto const it{ algorithms.find(name) };
    if (it == algorithms.end())
    {
        return fallback;
    }
    return it->second;
}


benchmark::Config benchmark::Config::makeDefault()
{
    Config cfg{};

    cfg.nvidia.enabled     = true;
    cfg.nvidia.deviceIndex = 0u;
    cfg.nvidia.algorithms["keccak"]             = { true, { 10u, 128u, 1024u }, {} };
    cfg.nvidia.algorithms["fnv1"]               = { true, { 10u, 1024u, 8192u }, {} };
    cfg.nvidia.algorithms["ethash_light_cache"] = { true, { 1u,  0u,    1u    }, {} };
    cfg.nvidia.algorithms["ethash"]             = { true, { 10u, 128u,  8192u }, {} };
    cfg.nvidia.algorithms["autolykos_v2"]       = { true, { 10u, 64u,   0u    }, {} };
    cfg.nvidia.algorithms["kawpow"]             = { true, { 10u, 256u,  1024u }, {} };

    cfg.amd.enabled     = true;
    cfg.amd.deviceIndex = 0u;
    cfg.amd.algorithms["kawpow"] = { true, { 1u, 256u, 1024u }, {} };

    return cfg;
}


// cppcheck-suppress unusedFunction
benchmark::Config benchmark::Config::loadFromFile(std::string const& path)
{
    Config cfg{ makeDefault() };

    std::ifstream file{ path };
    if (false == file.is_open())
    {
        logInfo() << "Benchmark config not found (" << path << "), using defaults";
        return cfg;
    }

    std::ostringstream ss{};
    ss << file.rdbuf();
    std::string const content{ ss.str() };

    try
    {
        logInfo() << "Benchmark config file: " << path;
        boost::json::value const root{ boost::json::parse(content) };
        auto const& obj{ root.as_object() };

        if (obj.contains("nvidia"))
        {
            cfg.nvidia = parseVendorConfig(obj.at("nvidia").as_object(), cfg.nvidia);
        }
        if (obj.contains("amd"))
        {
            cfg.amd = parseVendorConfig(obj.at("amd").as_object(), cfg.amd);
        }
    }
    catch (std::exception const& e)
    {
        logErr() << "Failed to parse benchmark config: " << e.what();
        logInfo() << "Using default config";
        return makeDefault();
    }

    return cfg;
}
