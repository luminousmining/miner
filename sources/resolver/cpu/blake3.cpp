#if defined(CPU_ENABLE)

#include <cstring>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>

#include <boost/json.hpp>

#include <algo/bitwise.hpp>
#include <algo/blake3/blake3_pow.hpp>
#include <algo/hash.hpp>
#include <common/cast.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <resolver/cpu/blake3.hpp>
#include <resolver/cpu/cpu_params.hpp>


// Alephium nonce widths in hex characters: the 8-byte big-endian search value occupies the low
// 16 hex chars; the pool recomputes the full 24-byte (48 hex char) nonce, so the submitted value
// is the search value zero-extended to the right.
constexpr uint32_t SEARCH_NONCE_HEX_LENGTH{ 16u };
constexpr uint32_t FULL_NONCE_HEX_LENGTH{ 48u };


resolver::ResolverCpuBlake3::PoolConfig resolver::ResolverCpuBlake3::resolvePoolConfig()
{
    common::Config const& config{ common::Config::instance() };

    // Parse the affinity mask exactly once: the worker-count resolution needs it (popcount
    // when --cpu_threads is unset) and the pool needs it for pinning.
    uint64_t const mask{ std::nullopt != config.cpu.affinity ? algo::hexToDecimal<uint64_t>(*config.cpu.affinity)
                                                             : 0ull };
    uint32_t const workers{
        resolver::cpu::resolveWorkerCount(config.cpu.threads, mask, std::thread::hardware_concurrency())
    };
    return PoolConfig{ workers, mask };
}


resolver::ResolverCpuBlake3::ResolverCpuBlake3(PoolConfig const poolConfig)
    : resolver::ResolverCpu(), threadPool{ poolConfig.workerCount, poolConfig.affinityMask }
{
    // Install the chunk callback once: it scans into whichever buffer currentIndexStream selects,
    // so executeSync()/executeAsync() only flip the index, never reassign the callback.
    threadPool.setCallback(
        [this](uint64_t const lo, uint64_t const hi, [[maybe_unused]] uint32_t const workerIndex)
        {
            hashChunk(lo, hi, batch[currentIndexStream]);
        });
}


resolver::ResolverCpuBlake3::ResolverCpuBlake3() : ResolverCpuBlake3(resolvePoolConfig())
{
    // Default CPU occupancy: DEFAULT_THREADS * DEFAULT_BLOCKS = 262144 nonces scanned per
    // executeSync(). Big enough to amortize the thread-pool dispatch per batch (tiny batches
    // measurably cut multi-core throughput), small enough that ~8 batches still complete
    // between a pool's frequent jobs so the hashrate displays (see
    // DeviceCpu::getMinimumKernelExecuted). Overridden by --threads / --blocks.
    constexpr uint32_t DEFAULT_THREADS{ 512u };
    constexpr uint32_t DEFAULT_BLOCKS{ 512u };

    algorithm = algo::ALGORITHM::BLAKE3;
    overrideOccupancy(DEFAULT_THREADS, DEFAULT_BLOCKS);
}


resolver::ResolverCpuBlake3::~ResolverCpuBlake3()
{
    // Drain any in-flight async batch before the buffers it scans are destroyed. threadPool is
    // declared before batch[], so it is destroyed last -- without this wait, a worker finishing
    // an in-flight batch during threadPool's join would read already-freed batch[] memory.
    if (true == inFlight)
    {
        threadPool.wait();
        inFlight = false;
    }
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


void resolver::ResolverCpuBlake3::prepareBatch(Batch& batch, stratum::StratumJobInfo const& jobInfo)
{
    // Copy by value: the worker closure for an async batch runs after executeAsync() returns,
    // so it cannot reference the caller's jobInfo.
    batch.header = jobInfo.headerBlob;
    batch.target = jobInfo.targetBlob;
    batch.base = jobInfo.nonce;
    batch.result.found = false;
    batch.result.count = 0u;
}


void resolver::ResolverCpuBlake3::hashChunk(uint64_t const lo, uint64_t const hi, Batch& batch)
{
    for (uint64_t i{ lo }; i < hi; ++i)
    {
        uint64_t const candidate{ batch.base + i };

        algo::hash256 digest{};
        algo::blake3::hashRef(batch.header, candidate, digest);

        // Winner: digest <= target, byte-wise from index 0 (matches the kernel).
        int const comparison{ std::memcmp(digest.ubytes, batch.target.ubytes, algo::LEN_HASH_256_WORD_8) };
        if (0 >= comparison)
        {
            // Hits are astronomically rare, so this lock has negligible contention.
            std::scoped_lock<std::mutex> const guard{ batch.hitMutex };
            if (algo::blake3::MAX_RESULT > batch.result.count)
            {
                batch.result.nonces[batch.result.count] = candidate;
                ++batch.result.count;
                batch.result.found = true;
            }
        }
    }
}


void resolver::ResolverCpuBlake3::harvest(Batch& batch, stratum::StratumJobInfo const& jobInfo)
{
    if (true == batch.result.found)
    {
        uint32_t const found{ common::max_limit(batch.result.count, algo::blake3::MAX_RESULT) };

        resultShare.found = true;
        resultShare.fromGroup = jobInfo.fromGroup;
        resultShare.toGroup = jobInfo.toGroup;
        resultShare.count = found;
        resultShare.jobId = jobInfo.jobIDStr;
        resultShare.extraNonceSize = jobInfo.extraNonceSize;

        for (uint32_t i{ 0u }; i < found; ++i)
        {
            resultShare.nonces[i] = batch.result.nonces[i];
        }

        batch.result.found = false;
        batch.result.count = 0u;
    }
}


bool resolver::ResolverCpuBlake3::executeSync(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // Drain any async batch left in flight so the buffer is free to reuse synchronously.
    if (true == inFlight)
    {
        threadPool.wait();
        inFlight = false;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint64_t const count{ castU64(getBlocks()) * castU64(getThreads()) };
    Batch&         current{ batch[currentIndexStream] };

    prepareBatch(current, jobInfo);

    // Fan the nonce batch across the pinned worker pool and block until it finishes: the
    // synchronous path is the one tests and debugging rely on.
    threadPool.run(count);

    ////////////////////////////////////////////////////////////////////////////
    harvest(current, jobInfo);

    return true;
}


bool resolver::ResolverCpuBlake3::executeAsync(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    uint64_t const count{ castU64(getBlocks()) * castU64(getThreads()) };

    // CPU mirror of the GPU two-stream double-buffer (see ResolverNvidiaBlake3::executeAsync):
    // wait for the batch dispatched on the previous call, harvest it, then swap buffers and
    // launch the next batch into the now-idle one and return immediately. While the device
    // reads/submits the harvested buffer and syncs with stratum, the workers keep computing.
    if (true == inFlight)
    {
        threadPool.wait();
        harvest(batch[currentIndexStream], jobInfo);
    }

    ////////////////////////////////////////////////////////////////////////////
    swapIndexStream();
    Batch& next{ batch[currentIndexStream] };
    prepareBatch(next, jobInfo);

    threadPool.runAsync(count);
    inFlight = true;

    return true;
}


void resolver::ResolverCpuBlake3::submit(stratum::Stratum* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i{ 0u }; i < resultShare.count; ++i)
            {
                // Left-pad the 8-byte search value to its 16 hex chars (leading zeros matter), then
                // right-extend with zeros to the full 24-byte nonce the pool recomputes.
                std::stringstream nonceHexa;
                nonceHexa << std::setw(SEARCH_NONCE_HEX_LENGTH) << std::setfill('0') << std::hex
                          << resultShare.nonces[i];

                std::string nonceStr{ nonceHexa.str() };

                while (nonceStr.size() < FULL_NONCE_HEX_LENGTH)
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
                nonceHexa << std::setw(SEARCH_NONCE_HEX_LENGTH) << std::setfill('0') << std::hex
                          << resultShare.nonces[i];

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
