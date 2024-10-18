#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <common/config.hpp>
#include <resolver/resolver.hpp>


void resolver::Resolver::setBlocks(uint32_t const newBlocks)
{
    blocks = newBlocks;
}


void resolver::Resolver::setThreads(uint32_t const newThreads)
{
    threads = newThreads;
}


uint32_t resolver::Resolver::getBlocks() const
{
    
    return blocks;
}


uint32_t resolver::Resolver::getThreads() const
{
    return threads;
}


void resolver::Resolver::updateJobId(
    std::string const& _jobId)
{
    jobId.assign(_jobId);
}


bool resolver::Resolver::isStale(
    std::string const& _jobId) const
{
    common::Config const& config { common::Config::instance() };

    if (false == config.mining.stale)
    {
        if (jobId != _jobId)
        {
            logWarn() << "Stale share detected, ignore it";
            return true;
        }
    }

    return false;
}
