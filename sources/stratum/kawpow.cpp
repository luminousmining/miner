#include <algo/ethash/ethash.hpp>
#include <algo/progpow/kawpow.hpp>
#include <stratum/kawpow.hpp>


stratum::StratumKawPOW::StratumKawPOW() :
    stratum::StratumProgPOW()
{
    maxPeriod = algo::kawpow::MAX_PERIOD;
    maxEpochLength = algo::progpow::EPOCH_LENGTH;
}
