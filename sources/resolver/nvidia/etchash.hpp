#pragma once

#if defined(CUDA_ENABLE)

#include <algo/dag_context.hpp>
#include <algo/hash.hpp>
#include <algo/ethash/result.hpp>
#include <resolver/nvidia/ethash.hpp>
#include <resolver/nvidia/ethash_kernel_parameter.hpp>


namespace resolver
{
    class ResolverNvidiaEtchash : public resolver::ResolverNvidiaEthash
    {
    public:
        ResolverNvidiaEtchash();
        ~ResolverNvidiaEtchash() = default;

    protected:
        bool updateContext(stratum::StratumJobInfo const& jobInfo) final;
    };
}

#endif
