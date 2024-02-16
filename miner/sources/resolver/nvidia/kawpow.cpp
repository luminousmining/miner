#include <algo/ethash/ethash.hpp>
#include <algo/progpow/kawpow.hpp>
#include <resolver/nvidia/kawpow.hpp>


resolver::ResolverNvidiaKawPOW::ResolverNvidiaKawPOW():
    resolver::ResolverNvidiaProgPOW()
{
    // Etash
    maxEpoch = algo::ethash::MAX_EPOCH_NUMBER;
    dagCountItemsGrowth = algo::ethash::DAG_COUNT_ITEMS_GROWTH;
    dagCountItemsInit = algo::ethash::DAG_COUNT_ITEMS_INIT;

    // KawPow
    progpowVersion = algo::progpow::VERSION::KAWPOW;
    dagItemParents = algo::kawpow::DAG_ITEM_PARENTS;
    countCache = algo::kawpow::COUNT_CACHE;
    countMath = algo::kawpow::COUNT_MATH;

    // Seed
    kernelSHA256.assign("kawpow_seed.cuh");
}
