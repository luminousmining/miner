#include <common/chrono.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/cast.hpp>
#include <common/log/log.hpp>
#include <device/device.hpp>
#include <resolver/amd/autolykos_v2.hpp>
#include <resolver/amd/etchash.hpp>
#include <resolver/amd/ethash.hpp>
#include <resolver/amd/evrprogpow.hpp>
#include <resolver/amd/firopow.hpp>
#include <resolver/amd/kawpow.hpp>
#include <resolver/amd/meowpow.hpp>
#include <resolver/amd/progpow_quai.hpp>
#include <resolver/amd/progpow.hpp>
#include <resolver/nvidia/autolykos_v2.hpp>
#include <resolver/nvidia/blake3.hpp>
#include <resolver/nvidia/etchash.hpp>
#include <resolver/nvidia/ethash.hpp>
#include <resolver/nvidia/evrprogpow.hpp>
#include <resolver/nvidia/firopow.hpp>
#include <resolver/nvidia/kawpow.hpp>
#include <resolver/nvidia/meowpow.hpp>
#include <resolver/nvidia/progpow_quai.hpp>
#include <resolver/nvidia/progpow.hpp>


void device::Device::setAlgorithm(
    algo::ALGORITHM newAlgorithm)
{
    ////////////////////////////////////////////////////////////////////////////
    common::Config const& config { common::Config::instance() };

    ////////////////////////////////////////////////////////////////////////////
    algorithm = newAlgorithm;
    switch (algorithm)
    {
        case algo::ALGORITHM::SHA256:
        {
            break;
        }
        case algo::ALGORITHM::ETHASH:
        {
            switch (deviceType)
            {
#if defined(CUDA_ENABLE)
                case device::DEVICE_TYPE::NVIDIA:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverNvidiaEthash);
                    break;
                }
#endif
#if defined(AMD_ENABLE)
                case device::DEVICE_TYPE::AMD:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverAmdEthash);
                    break;
                }
#endif
                case device::DEVICE_TYPE::UNKNOW:
                {
                    break;
                }
            }
            break;
        }
        case algo::ALGORITHM::ETCHASH:
        {
            switch (deviceType)
            {
#if defined(CUDA_ENABLE)
                case device::DEVICE_TYPE::NVIDIA:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverNvidiaEtchash);
                    break;
                }
#endif
#if defined(AMD_ENABLE)
                case device::DEVICE_TYPE::AMD:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverAmdEtchash);
                    break;
                }
#endif
                case device::DEVICE_TYPE::UNKNOW:
                {
                    break;
                }
            }
            break;
        }
        case algo::ALGORITHM::PROGPOW:
        {
            switch (deviceType)
            {
#if defined(CUDA_ENABLE)
                case device::DEVICE_TYPE::NVIDIA:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverNvidiaProgPOW);
                    break;
                }
#endif
#if defined(AMD_ENABLE)
                case device::DEVICE_TYPE::AMD:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverAmdProgPOW);
                    break;
                }
#endif
                case device::DEVICE_TYPE::UNKNOW:
                {
                    break;
                }
            }
            break;
        }
        case algo::ALGORITHM::KAWPOW:
        {
            switch (deviceType)
            {
#if defined(CUDA_ENABLE)
                case device::DEVICE_TYPE::NVIDIA:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverNvidiaKawPOW);
                    break;
                }
#endif
#if defined(AMD_ENABLE)
                case device::DEVICE_TYPE::AMD:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverAmdKawPOW);
                    break;
                }
#endif
                case device::DEVICE_TYPE::UNKNOW:
                {
                    break;
                }
            }
            break;
        }
        case algo::ALGORITHM::MEOWPOW:
        {
            switch (deviceType)
            {
#if defined(CUDA_ENABLE)
                case device::DEVICE_TYPE::NVIDIA:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverNvidiaMeowPOW);
                    break;
                }
#endif
#if defined(AMD_ENABLE)
                case device::DEVICE_TYPE::AMD:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverAmdMeowPOW);
                    break;
                }
#endif
                case device::DEVICE_TYPE::UNKNOW:
                {
                    break;
                }
            }
            break;
        }
        case algo::ALGORITHM::FIROPOW:
        {
            switch (deviceType)
            {
#if defined(CUDA_ENABLE)
                case device::DEVICE_TYPE::NVIDIA:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverNvidiaFiroPOW);
                    break;
                }
#endif
#if defined(AMD_ENABLE)
                case device::DEVICE_TYPE::AMD:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverAmdFiroPOW);
                    break;
                }
#endif
                case device::DEVICE_TYPE::UNKNOW:
                {
                    break;
                }
            }
            break;
        }
        case algo::ALGORITHM::EVRPROGPOW:
        {
            switch (deviceType)
            {
#if defined(CUDA_ENABLE)
                case device::DEVICE_TYPE::NVIDIA:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverNvidiaEvrprogPOW);
                    break;
                }
#endif
#if defined(AMD_ENABLE)
                case device::DEVICE_TYPE::AMD:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverAmdEvrprogPOW);
                    break;
                }
#endif
                case device::DEVICE_TYPE::UNKNOW:
                {
                    break;
                }
            }
            break;
        }
        case algo::ALGORITHM::PROGPOWQUAI:
        {
            switch (deviceType)
            {
#if defined(CUDA_ENABLE)
                case device::DEVICE_TYPE::NVIDIA:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverNvidiaProgpowQuai);
                    break;
                }
#endif
#if defined(AMD_ENABLE)
                case device::DEVICE_TYPE::AMD:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverAmdProgpowQuai);
                    break;
                }
#endif
                case device::DEVICE_TYPE::UNKNOW:
                {
                    break;
                }
            }
            break;
        }
        case algo::ALGORITHM::AUTOLYKOS_V2:
        {
            switch (deviceType)
            {
#if defined(CUDA_ENABLE)
                case device::DEVICE_TYPE::NVIDIA:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverNvidiaAutolykosV2);
                    break;
                }
#endif
#if defined(AMD_ENABLE)
                case device::DEVICE_TYPE::AMD:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverAmdAutolykosV2);
                    break;
                }
#endif
                case device::DEVICE_TYPE::UNKNOW:
                {
                    break;
                }
            }
            break;
        }
        case algo::ALGORITHM::BLAKE3:
        {
            switch (deviceType)
            {
#if defined(CUDA_ENABLE)
                case device::DEVICE_TYPE::NVIDIA:
                {
                    SAFE_DELETE(resolver);
                    resolver = NEW(resolver::ResolverNvidiaBlake3);
                    break;
                }
#endif
#if defined(AMD_ENABLE)
                case device::DEVICE_TYPE::AMD:
                {
                    break;
                }
#endif
                case device::DEVICE_TYPE::UNKNOW:
                {
                    break;
                }
            }
            break;
        }
        default:
        {
            kill(device::KILL_STATE::ALGORITH_UNDEFINED);
            return;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (nullptr == resolver)
    {
        kill(device::KILL_STATE::RESOLVER_NULLPTR);
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Set default value
    resolver->deviceId = id;
    resolver->deviceMemoryAvailable = memoryAvailable;
    if (std::nullopt != config.occupancy.blocks)
    {
        resolver->setBlocks(*config.occupancy.blocks);
    }
    if (std::nullopt != config.occupancy.threads)
    {
        resolver->setThreads(*config.occupancy.threads);
    }
}


void device::Device::setStratum(
    stratum::Stratum* const newStratum)
{
    stratum = newStratum;
}


void device::Device::setStratumSmartMining(
    stratum::StratumSmartMining* const newStratum)
{
    stratumSmartMining = newStratum;
}

void device::Device::kill(
    device::KILL_STATE const state)
{
    switch (state)
    {
        case device::KILL_STATE::ALGORITH_UNDEFINED:
        {
            deviceWarn()
                << "device[" << id << "] " << "Killed by code " << castU32(state)
                << " ALGORITH_UNDEFINED";
            break;
        }
        case device::KILL_STATE::RESOLVER_NULLPTR:
        {
            deviceWarn()
                << "device[" << id << "] " << "Killed by code " << castU32(state)
                << " RESOLVER_NULLPTR";
            break;
        }
        case device::KILL_STATE::UPDATE_MEMORY_FAIL:
        {
            deviceWarn()
                << "device[" << id << "] " << "Killed by code " << castU32(state)
                << " UPDATE_MEMORY_FAIL";
            break;
        }
        case device::KILL_STATE::UPDATE_CONSTANT_FAIL:
        {
            deviceWarn()
                << "device[" << id << "] " << "Killed by code " << castU32(state)
                << " UPDATE_CONSTANT_FAIL";
            break;
        }
        case device::KILL_STATE::KERNEL_EXECUTE_FAIL:
        {
            deviceWarn()
                << "device[" << id << "] " << "Killed by code " << castU32(state)
                << " KERNEL_EXECUTE_FAIL";
            break;
        }
        case device::KILL_STATE::DISABLE:
        {
            deviceWarn()
                << "device[" << id << "] " << "Killed by code " << castU32(state)
                << " DISABLE";
            break;
        }
    }
    alive.store(false, boost::memory_order::seq_cst);
}


bool device::Device::isAlive() const
{
    return alive.load(boost::memory_order::relaxed);
}


bool device::Device::isComputing() const
{
    return computing.load(boost::memory_order::relaxed);
}


void device::Device::update(
    bool const memory,
    bool const constants,
    stratum::StratumJobInfo const& newJobInfo)
{
    UNIQUE_LOCK(mtxUpdate);

    nextjobInfo = newJobInfo;
    nextjobInfo.nonce += (nextjobInfo.gapNonce * id);

    needUpdateMemory.store(memory, boost::memory_order::seq_cst);
    needUpdateConstants.store(constants, boost::memory_order::seq_cst);
}


void device::Device::increaseShare(
    bool const isValid)
{
    statistical::Statistical::ShareInfo& info{ miningStats.getShares() };
    ++info.total;
    if (true == isValid)
    {
        ++info.valid;
    }
    else
    {
        ++info.invalid;
    }
}


double device::Device::getHashrate()
{
    uint32_t const executeCount { miningStats.getKernelExecutedCount() };

    if (kernelMinimunExecuteNeeded <= executeCount)
    {
        miningStats.stop();
        miningStats.updateHashrate();
        miningStats.reset();
    }

    return miningStats.getHashrate();
}


stratum::Stratum* device::Device::getStratum()
{
    return stratum;
}


stratum::StratumSmartMining* device::Device::getStratumSmartMining()
{
    return stratumSmartMining;
}


statistical::Statistical::ShareInfo device::Device::getShare()
{
    statistical::Statistical::ShareInfo info{ miningStats.getShares() };
    return info;
}


void device::Device::run()
{
    threadDoWork.interrupt();
    threadDoWork = boost::thread{ boost::bind(&device::Device::loopDoWork, this) };
}


void device::Device::waitJob()
{
    while (nextjobInfo.epoch == -1)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}


void device::Device::cleanJob()
{
    stratum::StratumJobInfo clean{};
    nextjobInfo = clean;
}


bool device::Device::updateJob()
{
    common::Chrono chrono{};
    bool expectedMemory{ true };
    bool expectedConstants{ true };

    UNIQUE_LOCK(mtxUpdate);

    ////////////////////////////////////////////////////////////////////////////
    {
        needUpdateMemory.compare_exchange_weak(expectedMemory,
                                               false,
                                               boost::memory_order::seq_cst);
        needUpdateConstants.compare_exchange_weak(expectedConstants,
                                                  false,
                                                  boost::memory_order::seq_cst);
        if (   true == expectedMemory
            || true == expectedConstants)
        {
            if (   nextjobInfo.epoch != currentJobInfo.epoch
                || nextjobInfo.period != currentJobInfo.period)
            {
                miningStats.reset();
            }
            currentJobInfo = nextjobInfo;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == expectedMemory)
    {
        chrono.start();
        if (false == resolver->updateMemory(currentJobInfo))
        {
            kill(device::KILL_STATE::UPDATE_MEMORY_FAIL);
            return true;
        }
        chrono.stop();
        deviceInfo() << "Update memory in " << chrono.elapsed(common::CHRONO_UNIT::MS) << "ms";
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == expectedConstants)
    {
        chrono.start();
        resolver->updateJobId(currentJobInfo.jobIDStr);
        if (false == resolver->updateConstants(currentJobInfo))
        {
            kill(device::KILL_STATE::UPDATE_CONSTANT_FAIL);
            return true;
        }
        chrono.stop();
        deviceDebug() << "Update constants in " << chrono.elapsed(common::CHRONO_UNIT::US) << "us";
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == expectedMemory)
    {
        // Reset the stats, the memory was rebuilt
        miningStats.reset();
    }

    return (true == expectedMemory || true == expectedConstants);
}


void device::Device::loopDoWork()
{
    ////////////////////////////////////////////////////////////////////////////
    common::Config const& config { common::Config::instance() };
    if (false == config.isEnable(id))
    {
        kill(device::KILL_STATE::DISABLE);
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == initialize())
    {
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (nullptr == resolver)
    {
        deviceErr() << "Cannot works, device need resolver";
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    alive.store(true, boost::memory_order::relaxed);
    waitJob();

    ////////////////////////////////////////////////////////////////////////////
    // Statistical to compute the hashrate.
    miningStats.setBatchNonce(resolver->getBlocks() * resolver->getThreads());
    miningStats.resetHashrate();
    miningStats.reset();

    ////////////////////////////////////////////////////////////////////////////
    computing.store(true, boost::memory_order::seq_cst);

    ////////////////////////////////////////////////////////////////////////////
    while (   true == isAlive()
           && nullptr != resolver)
    {
        // Check and update the job.
        // Do not compute directly after update device.
        // A new job should spawn during the update.
        if (true == updateJob())
        {
            miningStats.setBatchNonce(resolver->getBlocks() * resolver->getThreads());
            continue;
        }

        // Execute the kernel to compute nonces.
        if (false == resolver->executeAsync(currentJobInfo))
        {
            kill(device::KILL_STATE::KERNEL_EXECUTE_FAIL);
            continue;
        }

        if (common::PROFILE::STANDARD == config.profile)
        {
            resolver->submit(stratum);
        }
        else
        {
            resolver->submit(stratumSmartMining);
        }

        miningStats.increaseKernelExecuted();

        // Increasing nonce to next kernel.
        currentJobInfo.nonce += miningStats.getBatchNonce();
    }

    ////////////////////////////////////////////////////////////////////////////
    cleanUp();

    ////////////////////////////////////////////////////////////////////////////
    computing.store(false, boost::memory_order::seq_cst);
}
