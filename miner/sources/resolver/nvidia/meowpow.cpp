#include <algo/ethash/ethash.hpp>
#include <algo/progpow/meowpow.hpp>
#include <resolver/nvidia/meowpow.hpp>


resolver::ResolverNvidiaMeowPOW::ResolverNvidiaMeowPOW():
    resolver::ResolverNvidiaProgPOW()
{
    // Etash
    maxEpoch = algo::ethash::MAX_EPOCH_NUMBER;
    dagCountItemsGrowth = algo::ethash::DAG_COUNT_ITEMS_GROWTH;
    dagCountItemsInit = algo::ethash::DAG_COUNT_ITEMS_INIT;

    // KawPow
    progpowVersion = algo::progpow::VERSION::KAWPOW;
    dagItemParents = algo::meowpow::DAG_ITEM_PARENTS;
    countCache = algo::meowpow::COUNT_CACHE;
    countMath = algo::meowpow::COUNT_MATH;
}
