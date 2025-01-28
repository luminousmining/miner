#include <algo/ethash/ethash.hpp>
#include <algo/progpow/progpow_quai.hpp>
#include <algo/hash_utils.hpp>
#include <common/boost_utils.hpp>
#include <common/custom.hpp>
#include <stratum/progpow_quai.hpp>


stratum::StratumProgpowQuai::StratumProgpowQuai() :
    stratum::StratumProgPOW()
{
    maxPeriod = algo::progpow_quai::MAX_PERIOD;
    maxEpochLength = algo::progpow_quai::EPOCH_LENGTH;
}
