#include <algo/ethash/ethash.hpp>
#include <algo/progpow/meowpow.hpp>
#include <stratum/meowpow.hpp>


stratum::StratumMeowPOW::StratumMeowPOW() :
    stratum::StratumProgPOW()
{
    maxPeriod = algo::meowpow::MAX_PERIOD;
    maxEpochLength = algo::progpow::EPOCH_LENGTH;
}
