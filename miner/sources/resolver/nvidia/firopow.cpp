#include <algo/ethash/ethash.hpp>
#include <algo/progpow/firopow.hpp>
#include <resolver/nvidia/firopow.hpp>


resolver::ResolverNvidiaFiroPOW::ResolverNvidiaFiroPOW():
    resolver::ResolverNvidiaProgPOW()
{
    // Etash
    maxEpoch = algo::ethash::MAX_EPOCH_NUMBER;
    dagCountItemsGrowth = algo::ethash::DAG_COUNT_ITEMS_GROWTH;
    dagCountItemsInit = algo::firopow::DAG_COUNT_ITEMS_INIT;

    // FiroPow
    dagItemParents = algo::firopow::DAG_ITEM_PARENTS;
    countCache = algo::firopow::COUNT_CACHE;
    countMath = algo::firopow::COUNT_MATH;

    // Seed
    kernelSHA256.assign("firopow_seed.cuh");
}
