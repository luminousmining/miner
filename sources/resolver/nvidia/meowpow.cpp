#include <algo/ethash/ethash.hpp>
#include <algo/progpow/meowpow.hpp>
#include <resolver/nvidia/meowpow.hpp>


resolver::ResolverNvidiaMeowPOW::ResolverNvidiaMeowPOW():
    resolver::ResolverNvidiaProgPOW()
{
    ///////////////////////////////////////////////////////////////////////////
    algorithm = algo::ALGORITHM::MEOWPOW;

    ///////////////////////////////////////////////////////////////////////////
    // Ethash
    maxEpoch = algo::ethash::EIP1057_MAX_EPOCH_NUMER;
    dagCountItemsGrowth = algo::ethash::DAG_COUNT_ITEMS_GROWTH;

    ///////////////////////////////////////////////////////////////////////////
    // ProgPow
    progpowVersion = algo::progpow::VERSION::MEOWPOW;
    regs = algo::meowpow::REGS;
    moduleSource = algo::meowpow::MODULE_SOURCE;

    ///////////////////////////////////////////////////////////////////////////
    // MeowPow
    dagItemParents = algo::meowpow::DAG_ITEM_PARENTS;
    countCache = algo::meowpow::COUNT_CACHE;
    countMath = algo::meowpow::COUNT_MATH;
}
