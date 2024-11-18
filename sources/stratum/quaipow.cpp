#include <algo/ethash/ethash.hpp>
#include <algo/progpow/quaipow.hpp>
#include <stratum/quaipow.hpp>


stratum::StratumQuaiPOW::StratumQuaiPOW() :
    stratum::StratumProgPOW()
{
    maxPeriod = algo::quaipow::MAX_PERIOD;
    maxEpochLength = algo::progpow::EPOCH_LENGTH;
}
