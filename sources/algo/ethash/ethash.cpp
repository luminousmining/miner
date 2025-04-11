#include <algo/bitwise.hpp>
#include <algo/keccak.hpp>
#include <algo/math.hpp>
#include <algo/ethash/ethash.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>


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
    algo::hash256  cached{ cachedSeed };

    ////////////////////////////////////////////////////////////////////////////
    if (seedpart == cached.word32[0])
    {
        return epochNumber;
    }

    ////////////////////////////////////////////////////////////////////////////
    cached = algo::keccak(cached);
    if (cached.word32[0] == seedpart)
    {
        cachedSeed = cached;
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


void algo::ethash::initializeDagContext(
    algo::DagContext& context,
    uint64_t const currentEpoch,
    uint32_t const maxEpoch,
    uint64_t const dagCountItemsGrowth,
    uint64_t const dagCountItemsInit,
    uint32_t const lightCacheCountItemsGrowth,
    uint32_t const lightCacheCountItemsInit)
{
    ////////////////////////////////////////////////////////////////////////////
    context.epoch = castU32(currentEpoch);
    if (   castU32(context.epoch) > maxEpoch
        && algo::ethash::EIP1057_MAX_EPOCH_NUMER != maxEpoch)
    {
        logErr() << "context.epoch: " << context.epoch << " | maxEpoch: " << maxEpoch;
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
    context.lightCache.numberItem = algo::largestPrime(lightCacheNumItemsUpperBound);
    context.lightCache.size = context.lightCache.numberItem * algo::LEN_HASH_512;

    ////////////////////////////////////////////////////////////////////////////
    uint64_t numberItemUpperBound{ castU64(epochEIP) };
    numberItemUpperBound *= dagCountItemsGrowth;
    numberItemUpperBound += dagCountItemsInit;
    context.dagCache.numberItem = algo::largestPrime(numberItemUpperBound);
    context.dagCache.size = context.dagCache.numberItem * algo::LEN_HASH_1024;

    ////////////////////////////////////////////////////////////////////////////
    size_t const dataLength{ algo::LEN_HASH_512 + context.lightCache.size };
    context.data = NEW(char[dataLength]{});
    if (nullptr == context.data)
    {
        logErr() << "Cannot alloc context data";
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    algo::hash256 seed{};
    for (int32_t i{ 0 }; i < context.epoch; ++i)
    {
        seed = algo::keccak(seed);
    }
    context.lightCache.hash = castPtrHash512(context.data + algo::LEN_HASH_512);

    ////////////////////////////////////////////////////////////////////////////
    buildLightCache(context, seed);
}


void algo::ethash::buildLightCache(
    algo::DagContext& context,
    algo::hash256 const& seed)
{
    ////////////////////////////////////////////////////////////////////////////
    algo::hash512 item{ algo::keccak<algo::hash512, algo::hash256>(seed) };
    context.lightCache.hash[0] = item;

    ////////////////////////////////////////////////////////////////////////////
    for (uint64_t i{ 1ull }; i < context.lightCache.numberItem; ++i)
    {
        item = algo::keccak(item);
        context.lightCache.hash[i] = item;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const numberItemu32 { castU32(context.lightCache.numberItem) };
    for (uint64_t round{ 0ull }; round < algo::ethash::LIGHT_CACHE_ROUNDS; ++round)
    {
        for (uint64_t i{ 0ull }; i < context.lightCache.numberItem; ++i)
        {
            uint32_t const fi{ context.lightCache.hash[i].word32[0] % numberItemu32 };
            uint32_t const si{ (numberItemu32 + (castU32(i) - 1u)) % numberItemu32 };

            algo::hash512 const& firstCache{ context.lightCache.hash[fi] };
            algo::hash512 const& secondCache{ context.lightCache.hash[si] };

            context.lightCache.hash[i] = algo::keccak(algo::hashXor(firstCache, secondCache));
        }
    }
}
