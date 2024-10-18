#include <algo/progpow/evrprogpow.hpp>
#include <stratum/evrprogpow.hpp>


stratum::StratumEvrprogPOW::StratumEvrprogPOW() :
    stratum::StratumProgPOW()
{
    maxPeriod = algo::evrprogpow::MAX_PERIOD;
    maxEpochLength = algo::evrprogpow::EPOCH_LENGTH;
}
