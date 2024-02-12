#pragma once

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
        ResolverNvidiaEtchash() = default;
        ~ResolverNvidiaEtchash() = default;

    protected:
        void updateContext(stratum::StratumJobInfo const& jobInfo) final;
    };
}
