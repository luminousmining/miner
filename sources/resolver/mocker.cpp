#if defined(TOOL_MOCKER)

#include <boost/chrono/duration.hpp>


#include <common/log/log.hpp>
#include <resolver/mocker.hpp>


static constexpr boost::chrono::milliseconds WAIT_UPDATE_MEMORY{ 40000 };
static constexpr boost::chrono::milliseconds WAIT_UPDATE_CONSTANT{ 2000 };
static constexpr boost::chrono::milliseconds WAIT_UPDATE_EXECUTE_SYNC{ 1000 };
static constexpr boost::chrono::milliseconds WAIT_UPDATE_EXECUTE_ASYNC{ 1000 };


void  resolver::ResolverMocker::overrideOccupancy(
    [[maybe_unused]] uint32_t const defaultThreads,
    [[maybe_unused]] uint32_t const defaultBlocks)
{
}


bool resolver::ResolverMocker::updateMemory(
    [[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    boost::this_thread::sleep_for(WAIT_UPDATE_MEMORY);
    return true;
}


bool resolver::ResolverMocker::updateConstants(
    [[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    boost::this_thread::sleep_for(WAIT_UPDATE_CONSTANT);
    return true;
}


bool resolver::ResolverMocker::executeSync(
    [[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    boost::this_thread::sleep_for(WAIT_UPDATE_EXECUTE_SYNC);
    return true;
}


bool resolver::ResolverMocker::executeAsync(
    [[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    boost::this_thread::sleep_for(WAIT_UPDATE_EXECUTE_ASYNC);
    return true;
}


void resolver::ResolverMocker::submit(
    [[maybe_unused]] stratum::Stratum* const stratum)
{
}


void resolver::ResolverMocker::submit(
    [[maybe_unused]] stratum::StratumSmartMining* const stratum)
{
}

#endif
