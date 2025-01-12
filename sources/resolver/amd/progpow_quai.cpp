#include <algo/progpow/progpow_quai.hpp>
#include <resolver/amd/progpow_quai.hpp>


resolver::ResolverAmdProgpowQuai::ResolverAmdProgpowQuai():
    resolver::ResolverAmdProgPOW()
{
    // ProgpowQuai
    progpowVersion = algo::progpow::VERSION::PROGPOWQUAI;
    dagItemParents = algo::progpow_quai::DAG_ITEM_PARENTS;
    countCache = algo::progpow_quai::COUNT_CACHE;
    countMath = algo::progpow_quai::COUNT_MATH;
}
