#pragma once

#if defined(TOOL_MOCKER)

#include <resolver/resolver.hpp>


namespace resolver
{
    struct ResolverMocker : public resolver::Resolver
    {
    public:
        ~ResolverMocker() = default;

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;


    protected:
        void overrideOccupancy(uint32_t const defaultThreads,
                                    uint32_t const defaultBlocks) final;
    };
}

#endif
