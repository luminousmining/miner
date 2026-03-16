#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>


namespace benchmark
{
    struct KernelParams
    {
        uint32_t loop{ 10u };
        uint32_t threads{ 256u };
        uint32_t blocks{ 1024u };
    };

    struct AlgoConfig
    {
        bool                                              enabled{ true };
        KernelParams                                      defaults{};
        std::vector<std::pair<std::string, KernelParams>> kernels{};

        bool         isKernelEnabled(std::string const& name) const;
        KernelParams resolveKernel(std::string const& name) const;
    };

    struct VendorConfig
    {
        bool                              enabled{ true };
        uint32_t                          deviceIndex{ 0u };
        std::map<std::string, AlgoConfig> algorithms{};

        bool              isAlgoEnabled(std::string const& name) const;
        AlgoConfig const& getAlgo(std::string const& name) const;
    };

    struct Config
    {
        VendorConfig nvidia{};
        VendorConfig amd{};

        static Config loadFromFile(std::string const& path);
        static Config makeDefault();
    };
}
