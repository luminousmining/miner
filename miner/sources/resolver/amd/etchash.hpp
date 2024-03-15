#pragma once

#include <algo/dag_context.hpp>
#include <algo/hash.hpp>
#include <algo/ethash/result.hpp>
#include <common/kernel_generator.hpp>
#include <resolver/amd/ethash.hpp>
#include <resolver/amd/ethash_kernel_parameter.hpp>


namespace resolver
{
    class ResolverAmdEtchash : public resolver::ResolverAmdEthash
    {
    public:
        ResolverAmdEtchash() = default;
        ~ResolverAmdEtchash() = default;

    protected:
        bool updateContext(stratum::StratumJobInfo const& jobInfo) final;
    };
}
