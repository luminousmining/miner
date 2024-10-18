#include <stratum/etchash.hpp>


stratum::StratumEtchash::StratumEtchash() :
    stratum::StratumEthash()
{
    maxEpochNumber = algo::ethash::EIP1099_MAX_EPOCH_NUMBER;
}
