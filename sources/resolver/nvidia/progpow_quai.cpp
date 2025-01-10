#include <algo/progpow/progpow_quai.hpp>
#include <resolver/nvidia/progpow_quai.hpp>


resolver::ResolverNvidiaProgpowQuai::ResolverNvidiaProgpowQuai():
    resolver::ResolverNvidiaProgPOW()
{
    // ProgpowQuai
    dagItemParents = algo::progpow_quai::DAG_ITEM_PARENTS;
    countCache = algo::progpow_quai::COUNT_CACHE;
    countMath = algo::progpow_quai::COUNT_MATH;
}
