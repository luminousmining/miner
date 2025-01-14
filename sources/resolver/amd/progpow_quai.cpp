#include <algo/progpow/progpow_quai.hpp>
#include <resolver/amd/progpow_quai.hpp>


resolver::ResolverAmdProgpowQuai::ResolverAmdProgpowQuai():
    resolver::ResolverAmdProgPOW()
{
    //Ethash
    dagItemParents = algo::progpow_quai::DAG_ITEM_PARENTS;

    // ProgpowQuai
    progpowVersion = algo::progpow::VERSION::PROGPOWQUAI;
    countCache = algo::progpow_quai::COUNT_CACHE;
    countMath = algo::progpow_quai::COUNT_MATH;
}
