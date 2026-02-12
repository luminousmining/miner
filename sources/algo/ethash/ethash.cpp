#include <algo/bitwise.hpp>
#include <algo/hash_utils.hpp>
#include <algo/keccak.hpp>
#include <algo/math.hpp>
#include <algo/ethash/ethash.hpp>
#include <common/cast.hpp>
#include <common/config.hpp>
#include <common/chrono.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>


boost::mutex mtxDagContext{};
algo::DagContext localDagContext{};


int32_t algo::ethash::findEpoch(
    algo::hash256 const& seedHash,
    uint32_t const maxEpoch)
{
    ////////////////////////////////////////////////////////////////////////////
    static thread_local int32_t       cachedEpochNumber{ 0 };
    static thread_local algo::hash256 cachedSeed{};

    ////////////////////////////////////////////////////////////////////////////
    int32_t const  epochNumber{ cachedEpochNumber };
    uint32_t const seedpart{ seedHash.word32[0] };
    algo::hash256  seed{ cachedSeed };

    ////////////////////////////////////////////////////////////////////////////
    if (seedpart == seed.word32[0])
    {
        return epochNumber;
    }

    ////////////////////////////////////////////////////////////////////////////
    seed = algo::keccak(seed);
    if (seed.word32[0] == seedpart)
    {
        cachedSeed = seed;
        cachedEpochNumber = epochNumber + 1;
        return epochNumber + 1;
    }

    ////////////////////////////////////////////////////////////////////////////
    algo::hash256 hash{};
    for (uint32_t i{ 0u }; i <= maxEpoch; ++i)
    {
        if (hash.word32[0] == seedpart)
        {
            cachedSeed = hash;
            cachedEpochNumber = i;
            return i;
        }
        hash = algo::keccak(hash);
    }

    ////////////////////////////////////////////////////////////////////////////
    return -1;
}


void algo::ethash::freeDagContext(algo::DagContext& context)
{
    ////////////////////////////////////////////////////////////////////////////
    SAFE_DELETE_ARRAY(localDagContext.data);
    localDagContext.lightCache.hash = nullptr;
}


void algo::ethash::buildContext(
    algo::DagContext& context,
    uint64_t const currentEpoch,
    uint32_t const maxEpoch,
    uint64_t const dagCountItemsGrowth,
    uint64_t const dagCountItemsInit,
    uint32_t const lightCacheCountItemsGrowth,
    uint32_t const lightCacheCountItemsInit)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDagContext);

    ///////////////////////////////////////////////////////////////////////////
    common::Config& config{ common::Config::instance() };

    ////////////////////////////////////////////////////////////////////////////
    if (localDagContext.epoch == castU32(currentEpoch))
    {
        logInfo() << "Skip epoch: " << currentEpoch;
        copyContext(context);
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    localDagContext.epoch = castU32(currentEpoch);
    if (   castU32(localDagContext.epoch) > maxEpoch
        && algo::ethash::EIP1057_MAX_EPOCH_NUMER != maxEpoch)
    {
        logErr() << "context.epoch: " << localDagContext.epoch << " | maxEpoch: " << maxEpoch;
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint64_t epochEIP{ currentEpoch };
    if (algo::ethash::EIP1099_MAX_EPOCH_NUMBER == maxEpoch)
    {
        epochEIP /= 2ull;
    }
    else if (algo::ethash::EIP1057_MAX_EPOCH_NUMER == maxEpoch)
    {
        epochEIP *= 4ull;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint64_t lightCacheNumItemsUpperBound { castU64(epochEIP) };
    lightCacheNumItemsUpperBound *= lightCacheCountItemsGrowth;
    lightCacheNumItemsUpperBound += lightCacheCountItemsInit;
    localDagContext.lightCache.numberItem = algo::largestPrime(lightCacheNumItemsUpperBound);
    localDagContext.lightCache.size = localDagContext.lightCache.numberItem * algo::LEN_HASH_512;

    ////////////////////////////////////////////////////////////////////////////
    uint64_t numberItemUpperBound{ castU64(epochEIP) };
    numberItemUpperBound *= dagCountItemsGrowth;
    numberItemUpperBound += dagCountItemsInit;
    localDagContext.dagCache.numberItem = algo::largestPrime(numberItemUpperBound);
    localDagContext.dagCache.size = localDagContext.dagCache.numberItem * algo::LEN_HASH_1024;

    ////////////////////////////////////////////////////////////////////////////
    algo::hash256 seed{};
    for (int32_t i{ 0 }; i < localDagContext.epoch; ++i)
    {
        seed = algo::keccak(seed);
    }

    ////////////////////////////////////////////////////////////////////////////
    algo::hash512 seedHash{ algo::keccak<algo::hash512, algo::hash256>(seed) };
    algo::copyHash(localDagContext.hashedSeedCache, seedHash);

    ////////////////////////////////////////////////////////////////////////////
    if (true == config.deviceAlgorithm.ethashBuildLightCacheCPU)
    {
        algo::ethash::buildLightCacheOnCPU(context);
    }

    ////////////////////////////////////////////////////////////////////////////
    copyContext(context);
}


void algo::ethash::buildLightCacheOnCPU(
    algo::DagContext& context)
{
    ////////////////////////////////////////////////////////////////////////////
    algo::hash512 item{ localDagContext.hashedSeedCache };
    size_t const dataLength{ algo::LEN_HASH_512 + localDagContext.lightCache.size };
    localDagContext.data = NEW_ARRAY(char, dataLength);
    std::memset(localDagContext.data, 0, sizeof(char) * dataLength);
    if (nullptr == localDagContext.data)
    {
        logErr() << "Cannot alloc context data";
        return;
    }
    localDagContext.lightCache.hash = castPtrHash512(localDagContext.data + algo::LEN_HASH_512);

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Building light cache on CPU";
    common::ChronoGuard chrono{ "Built light cache", common::CHRONO_UNIT::MS };

    ////////////////////////////////////////////////////////////////////////////
    for (uint64_t i{ 0ull }; i < localDagContext.lightCache.numberItem; ++i)
    {
        localDagContext.lightCache.hash[i] = item;
        item = algo::keccak(item);
    }

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const numberItemu32{ castU32(localDagContext.lightCache.numberItem) };
    for (uint64_t round{ 0ull }; round < algo::ethash::LIGHT_CACHE_ROUNDS; ++round)
    {
        for (uint64_t i{ 0ull }; i < localDagContext.lightCache.numberItem; ++i)
        {
            uint32_t const fi{ localDagContext.lightCache.hash[i].word32[0] % numberItemu32 };
            uint32_t const si{ (numberItemu32 + (castU32(i) - 1u)) % numberItemu32 };

            algo::hash512 const& firstCache{ localDagContext.lightCache.hash[fi] };
            algo::hash512 const& secondCache{ localDagContext.lightCache.hash[si] };

            algo::hash512 const xored = algo::hashXor(firstCache, secondCache);
            localDagContext.lightCache.hash[i] = algo::keccak(xored);
        }
    }
}


void algo::ethash::copyContext(algo::DagContext& context)
{
    context.epoch = localDagContext.epoch;
    context.lightCache.numberItem = localDagContext.lightCache.numberItem;
    context.lightCache.size = localDagContext.lightCache.size;
    context.dagCache.numberItem = localDagContext.dagCache.numberItem;
    context.dagCache.size = localDagContext.dagCache.size;
    context.data = localDagContext.data;
    context.lightCache.hash = localDagContext.lightCache.hash;

    algo::copyHash(context.hashedSeedCache, localDagContext.hashedSeedCache);
}

