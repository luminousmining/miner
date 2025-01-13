#include <algo/progpow/progpow_quai.hpp>
#include <resolver/nvidia/progpow_quai.hpp>

#include <common/log/log.hpp>

resolver::ResolverNvidiaProgpowQuai::ResolverNvidiaProgpowQuai():
    resolver::ResolverNvidiaProgPOW()
{
    //Ethash
    dagItemParents = algo::progpow_quai::DAG_ITEM_PARENTS;

    // ProgpowQuai
    progpowVersion = algo::progpow::VERSION::PROGPOWQUAI;
    countCache = algo::progpow_quai::COUNT_CACHE;
    countMath = algo::progpow_quai::COUNT_MATH;
}

