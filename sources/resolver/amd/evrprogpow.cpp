#include <algo/ethash/ethash.hpp>
#include <algo/progpow/evrprogpow.hpp>
#include <resolver/amd/evrprogpow.hpp>


resolver::ResolverAmdEvrprogPOW::ResolverAmdEvrprogPOW():
    resolver::ResolverAmdProgPOW()
{
    ///////////////////////////////////////////////////////////////////////////
    algorithm = algo::ALGORITHM::EVRPROGPOW;

    ///////////////////////////////////////////////////////////////////////////
    // Ethash
    maxEpoch = algo::ethash::MAX_EPOCH_NUMBER;
    dagCountItemsGrowth = algo::ethash::DAG_COUNT_ITEMS_GROWTH;
    dagCountItemsInit = algo::evrprogpow::DAG_COUNT_ITEMS_INIT;

    ///////////////////////////////////////////////////////////////////////////
    // KawPow
    progpowVersion = algo::progpow::VERSION::EVRPROGPOW;
    dagItemParents = algo::evrprogpow::DAG_ITEM_PARENTS;
    countCache = algo::evrprogpow::COUNT_CACHE;
    countMath = algo::evrprogpow::COUNT_MATH;
}
