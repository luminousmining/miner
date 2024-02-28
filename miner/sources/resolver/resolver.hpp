#pragma once

#include <stratum/smart_mining.hpp>
#include <stratum/stratum.hpp>
#include <stratum/job_info.hpp>


namespace resolver
{
    class Resolver
    {
    public:
        uint32_t    deviceId { 0u };
        std::string jobId{};

        Resolver() = default;
        virtual ~Resolver() = default;

        Resolver(Resolver const&) = delete;
        Resolver(Resolver&&) = delete;
        Resolver& operator=(Resolver const&) = delete;
        Resolver& operator=(Resolver&&) = delete;

        void setBlocks(uint32_t const newBlocks);
        void setThreads(uint32_t const newThreads);
        uint32_t getBlocks() const;
        uint32_t getThreads() const;
        void updateJobId(std::string const& _jobId);

        virtual bool updateMemory(stratum::StratumJobInfo const& jobInfo) = 0;
        virtual bool updateConstants(stratum::StratumJobInfo const& jobInfo) = 0;
        virtual bool execute(stratum::StratumJobInfo const& jobInfo) = 0;
        virtual void submit(stratum::Stratum* const stratum) = 0;
        virtual void submit(stratum::StratumSmartMining* const stratum) = 0;

    protected:
        uint32_t blocks{ 1024u };
        uint32_t threads { 1024u };

        bool isStale(std::string const& _jobInfo) const;
    };
}
