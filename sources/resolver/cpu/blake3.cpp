#if defined(CPU_ENABLE)

#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>

#include <boost/json.hpp>

#include <algo/blake3/blake3_pow.hpp>
#include <algo/hash.hpp>
#include <common/cast.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <resolver/cpu/blake3.hpp>
#include <resolver/cpu/cpu_params.hpp>


// Default CPU occupancy: DEFAULT_THREADS * DEFAULT_BLOCKS = 262144 nonces scanned per
// executeSync(). Big enough to amortize the thread-pool dispatch per batch (tiny batches
// measurably cut multi-core throughput), small enough that ~8 batches still complete
// between a pool's frequent jobs so the hashrate displays (see
// DeviceCpu::getMinimumKernelExecuted). Overridden by --threads / --blocks.
namespace
{
    constexpr uint32_t DEFAULT_THREADS{ 512u };
    constexpr uint32_t DEFAULT_BLOCKS{ 512u };
}


resolver::ResolverCpuBlake3::PoolConfig resolver::ResolverCpuBlake3::resolvePoolConfig()
{
    common::Config const& config{ common::Config::instance() };

    // Parse the affinity mask exactly once: the worker-count resolution needs it (popcount
    // when --cpu_threads is unset) and the pool needs it for pinning.
    uint64_t const mask{ config.cpu.affinity.has_value() ? resolver::cpu_detail::parseHexMask(*config.cpu.affinity)
                                                         : 0ull };
    uint32_t const workers{
        resolver::cpu_detail::resolveWorkerCount(config.cpu.threads, mask, std::thread::hardware_concurrency())
    };
    return PoolConfig{ workers, mask };
}


resolver::ResolverCpuBlake3::ResolverCpuBlake3(PoolConfig const poolConfig)
    : resolver::ResolverCpu(), pool{ poolConfig.workerCount, poolConfig.affinityMask }
{
}


resolver::ResolverCpuBlake3::ResolverCpuBlake3() : ResolverCpuBlake3(resolvePoolConfig())
{
    algorithm = algo::ALGORITHM::BLAKE3;
    overrideOccupancy(DEFAULT_THREADS, DEFAULT_BLOCKS);
}


resolver::ResolverCpuBlake3::~ResolverCpuBlake3()
{
}


bool resolver::ResolverCpuBlake3::updateMemory([[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    // Blake3 has no DAG and no light cache: nothing to allocate.
    return true;
}


bool resolver::ResolverCpuBlake3::updateConstants([[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    // Header and target are read straight from jobInfo at execute time: nothing to precompute.
    return true;
}


bool resolver::ResolverCpuBlake3::executeSync(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    uint64_t const base{ jobInfo.nonce };
    uint64_t const count{ castU64(getBlocks()) * castU64(getThreads()) };
    algo::hash3072 header{ jobInfo.headerBlob };

    ////////////////////////////////////////////////////////////////////////////
    algo::blake3::Result local{ false, 0u, { 0ull, 0ull, 0ull, 0ull } };
    std::mutex           hitMutex{};

    // Fan the nonce batch across the pinned worker pool. Hits are astronomically rare, so
    // the shared result append is guarded by a mutex with negligible contention.
    pool.parallelFor(
        count,
        [&](uint64_t const lo, uint64_t const hi, uint32_t const /*workerIndex*/)
        {
            for (uint64_t i{ lo }; i < hi; ++i)
            {
                uint64_t const candidate{ base + i };

                algo::hash256 digest{};
                algo::blake3::hashRef(header, candidate, digest);

                // Winner: digest <= targetBlob, byte-wise from index 0 (matches the kernel).
                if (std::memcmp(digest.ubytes, jobInfo.targetBlob.ubytes, algo::LEN_HASH_256_WORD_8) <= 0)
                {
                    std::scoped_lock<std::mutex> const guard{ hitMutex };
                    if (local.count < algo::blake3::MAX_RESULT)
                    {
                        local.nonces[local.count] = candidate;
                        ++local.count;
                        local.found = true;
                    }
                }
            }
        });

    ////////////////////////////////////////////////////////////////////////////
    if (true == local.found)
    {
        uint32_t const found{ common::max_limit(local.count, algo::blake3::MAX_RESULT) };

        resultShare.found = true;
        resultShare.fromGroup = jobInfo.fromGroup;
        resultShare.toGroup = jobInfo.toGroup;
        resultShare.count = found;
        resultShare.jobId = jobInfo.jobIDStr;
        resultShare.extraNonceSize = jobInfo.extraNonceSize;

        for (uint32_t i{ 0u }; i < found; ++i)
        {
            resultShare.nonces[i] = local.nonces[i];
        }
    }

    return true;
}


bool resolver::ResolverCpuBlake3::executeAsync(stratum::StratumJobInfo const& jobInfo)
{
    // CPU work is synchronous: no double-buffering. Async == sync.
    return executeSync(jobInfo);
}


void resolver::ResolverCpuBlake3::submit(stratum::Stratum* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i{ 0u }; i < resultShare.count; ++i)
            {
                // Zero-pad to 16 hex (8 bytes): the pool recomputes the 24-byte nonce, so leading zeros matter.
                std::stringstream nonceHexa;
                nonceHexa << std::setw(16) << std::setfill('0') << std::hex << resultShare.nonces[i];

                std::string nonceStr{ nonceHexa.str() };

                while (nonceStr.size() < 48)
                {
                    nonceStr += "0";
                }

                boost::json::object params{};
                params["jobId"] = resultShare.jobId;
                params["fromGroup"] = resultShare.fromGroup;
                params["toGroup"] = resultShare.toGroup;
                params["nonce"] = nonceStr;

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }
    }

    resultShare.count = 0u;
    resultShare.found = false;
}


void resolver::ResolverCpuBlake3::submit(stratum::StratumSmartMining* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i{ 0u }; i < resultShare.count; ++i)
            {
                std::stringstream nonceHexa;
                nonceHexa << std::setw(16) << std::setfill('0') << std::hex << resultShare.nonces[i];

                boost::json::object params{};
                params["jobId"] = resultShare.jobId;
                params["fromGroup"] = resultShare.fromGroup;
                params["toGroup"] = resultShare.toGroup;
                params["nonce"] = nonceHexa.str().substr(resultShare.extraNonceSize);

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }
    }

    resultShare.count = 0u;
    resultShare.found = false;
}

#endif
