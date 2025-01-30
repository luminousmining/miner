#include <algo/ethash/ethash.hpp>
#include <algo/progpow/progpow_quai.hpp>
#include <algo/hash_utils.hpp>
#include <common/boost_utils.hpp>
#include <common/custom.hpp>
#include <stratum/progpow_z.hpp>


stratum::StratumProgpowZ::StratumProgpowZ() :
    stratum::StratumProgPOW()
{
    maxEpochLength = algo::ethash::EPOCH_LENGTH;
}
