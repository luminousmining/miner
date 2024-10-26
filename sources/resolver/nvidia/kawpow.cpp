#include <algo/ethash/ethash.hpp>
#include <algo/progpow/kawpow.hpp>
#include <resolver/nvidia/kawpow.hpp>


resolver::ResolverNvidiaKawPOW::ResolverNvidiaKawPOW():
    resolver::ResolverNvidiaProgPOW()
{
    // Ethash
    maxEpoch = algo::ethash::MAX_EPOCH_NUMBER;
    dagCountItemsGrowth = algo::ethash::DAG_COUNT_ITEMS_GROWTH;
    dagCountItemsInit = algo::ethash::DAG_COUNT_ITEMS_INIT;

    // ProgPow
    progpowVersion = algo::progpow::VERSION::KAWPOW;

    // KawPow
    dagItemParents = algo::kawpow::DAG_ITEM_PARENTS;
    countCache = algo::kawpow::COUNT_CACHE;
    countMath = algo::kawpow::COUNT_MATH;
}
