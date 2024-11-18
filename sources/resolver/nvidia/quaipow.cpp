#include <algo/ethash/ethash.hpp>
#include <algo/progpow/quaipow.hpp>
#include <resolver/nvidia/quaipow.hpp>


resolver::ResolverNvidiaQuaiPOW::ResolverNvidiaQuaiPOW():
    resolver::ResolverNvidiaProgPOW()
{
    // Ethash
    maxEpoch = algo::ethash::MAX_EPOCH_NUMBER;
    dagCountItemsGrowth = algo::ethash::DAG_COUNT_ITEMS_GROWTH;
    dagCountItemsInit = algo::ethash::DAG_COUNT_ITEMS_INIT;

    // ProgPow
    progpowVersion = algo::progpow::VERSION::QUAIPOW;

    // QuaiPow
    dagItemParents = algo::quaipow::DAG_ITEM_PARENTS;
    countCache = algo::quaipow::COUNT_CACHE;
    countMath = algo::quaipow::COUNT_MATH;
}
