#include <algo/progpow/firopow.hpp>
#include <stratum/firopow.hpp>


stratum::StratumFiroPOW::StratumFiroPOW() :
    stratum::StratumProgPOW()
{
    maxPeriod = algo::firopow::MAX_PERIOD;
    maxEpochLength = algo::firopow::EPOCH_LENGTH;
}
