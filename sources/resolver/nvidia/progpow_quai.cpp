#include <algo/progpow/progpow_quai.hpp>
#include <resolver/nvidia/progpow_quai.hpp>


resolver::ResolverNvidiaProgpowQuai::ResolverNvidiaProgpowQuai():
    resolver::ResolverNvidiaProgPOW()
{
    ///////////////////////////////////////////////////////////////////////////
    algorithm = algo::ALGORITHM::PROGPOWQUAI;

    //Ethash
    dagItemParents = algo::progpow_quai::DAG_ITEM_PARENTS;
    dagCountItemsGrowth = algo::progpow_quai::DAG_COUNT_ITEMS_GROWTH;
    dagCountItemsInit = algo::progpow_quai::DAG_COUNT_ITEMS_INIT;
    lightCacheCountItemsGrowth = algo::progpow_quai::LIGHT_CACHE_COUNT_ITEMS_GROWTH;

    // ProgpowQuai
    progpowVersion = algo::progpow::VERSION::PROGPOWQUAI;
    countCache = algo::progpow_quai::COUNT_CACHE;
    countMath = algo::progpow_quai::COUNT_MATH;
}
