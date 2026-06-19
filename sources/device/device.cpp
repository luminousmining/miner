#include <common/cast.hpp>
#include <common/chrono.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <device/device.hpp>
#include <resolver/mocker.hpp>


void device::Device::setAlgorithm(algo::ALGORITHM newAlgorithm)
{
    ////////////////////////////////////////////////////////////////////////////
    common::Config const& config{ common::Config::instance() };

    ////////////////////////////////////////////////////////////////////////////
    algorithm = newAlgorithm;
    switch (deviceType)
    {
#if defined(TOOL_MOCKER)
        case device::DEVICE_TYPE::MOCKER:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverMocker);
            break;
        }
#endif
#if defined(CUDA_ENABLE)
        case device::DEVICE_TYPE::NVIDIA:
        {
            setResolverNvidia(algorithm);
            break;
        }
#endif
#if defined(AMD_ENABLE)
        case device::DEVICE_TYPE::AMD:
        {
            setResolverAmd(algorithm);
            break;
        }
#endif
#if defined(CPU_ENABLE)
        case device::DEVICE_TYPE::CPU:
        {
            setResolverCpu(algorithm);
            break;
        }
#endif
        case device::DEVICE_TYPE::UNKNOWN:
        {
            break;
        }
        default:
        {
            kill(device::KILL_STATE::ALGORITH_UNDEFINED);
            return;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (nullptr == resolver) [[unlikely]]
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


void device::Device::setStratum(stratum::Stratum* const newStratum)
{
    stratum = newStratum;
}


void device::Device::setStratumSmartMining(stratum::StratumSmartMining* const newStratum)
{
    stratumSmartMining = newStratum;
}


void device::Device::kill(device::KILL_STATE const state)
{
    switch (state)
    {
        case device::KILL_STATE::ALGORITH_UNDEFINED:
        {
            deviceWarn() << "device[" << id << "] "
                         << "Killed by code " << castU32(state) << " ALGORITH_UNDEFINED";
            break;
        }
        case device::KILL_STATE::RESOLVER_NULLPTR:
        {
            deviceWarn() << "device[" << id << "] "
                         << "Killed by code " << castU32(state) << " RESOLVER_NULLPTR";
            break;
        }
        case device::KILL_STATE::UPDATE_MEMORY_FAIL:
        {
            deviceWarn() << "device[" << id << "] "
                         << "Killed by code " << castU32(state) << " UPDATE_MEMORY_FAIL";
            break;
        }
        case device::KILL_STATE::UPDATE_CONSTANT_FAIL:
        {
            deviceWarn() << "device[" << id << "] "
                         << "Killed by code " << castU32(state) << " UPDATE_CONSTANT_FAIL";
            break;
        }
        case device::KILL_STATE::KERNEL_EXECUTE_FAIL:
        {
            deviceWarn() << "device[" << id << "] "
                         << "Killed by code " << castU32(state) << " KERNEL_EXECUTE_FAIL";
            break;
        }
        case device::KILL_STATE::DISABLE:
        {
            deviceWarn() << "device[" << id << "] "
                         << "Killed by code " << castU32(state) << " DISABLE";
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


void device::Device::update(bool const memory, bool const constants, stratum::StratumJobInfo const& newJobInfo)
{
    nextjobInfo.copy(newJobInfo);
    nextjobInfo.nonce += (nextjobInfo.gapNonce * id);

    if (true == constants)
    {
        synchronizer.constant.add(1ull);
    }
    if (true == memory)
    {
        synchronizer.memory.add(1ull);
    }
    synchronizer.job.add(1ull);
}


void device::Device::increaseShare(bool const isValid)
{
    statistical::Statistical::ShareInfo& info{ miningStats.getShares() };
    ++info.total;
    if (true == isValid)
    {
        deviceInfo() << "Share valid";
        ++info.valid;
    }
    else
    {
        deviceErr() << "Share invalid";
        ++info.invalid;
    }
}


uint32_t device::Device::getMinimumKernelExecuted() const
{
    common::Config const& config{ common::Config::instance() };
    return config.occupancy.kernelMinimunExecuteNeeded;
}


double device::Device::getHashrate()
{
    uint32_t const executeCount{ miningStats.getKernelExecutedCount() };

    if (getMinimumKernelExecuted() <= executeCount)
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
    deviceDebug() << "waiting job!";
    while (true == synchronizer.job.isEqual())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}


void device::Device::cleanJob()
{
    stratum::StratumJobInfo clean{};
    nextjobInfo.copy(clean);
}


bool device::Device::updateJob()
{
    ////////////////////////////////////////////////////////////////////////////
    common::Chrono chrono{};
    bool const     needUpdateJob{ synchronizer.job.isEqual() == false ? true : false };
    bool const     needUpdateConstant{ synchronizer.constant.isEqual() == false ? true : false };
    bool const     needUpdateMemory{ synchronizer.memory.isEqual() == false ? true : false };

    ////////////////////////////////////////////////////////////////////////////
    if (false == needUpdateJob)
    {
        return false;
    }
    if (false == needUpdateConstant && false == needUpdateMemory)
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint64_t const currentAtomicJob{ synchronizer.job.get() };
    uint64_t const currentAtomicConstant{ synchronizer.constant.get() };
    uint64_t const currentAtomicMemory{ synchronizer.memory.get() };

    ////////////////////////////////////////////////////////////////////////////
    common::Config const& config{ common::Config::instance() };
    if (nextjobInfo.epoch != currentJobInfo.epoch || nextjobInfo.period != currentJobInfo.period)
    {
        if (false == config.occupancy.accumulateHash)
        {
            miningStats.reset();
        }
    }
    currentJobInfo.copy(nextjobInfo);
    synchronizer.job.update(currentAtomicJob);

    ////////////////////////////////////////////////////////////////////////////
    if (true == needUpdateMemory)
    {
        synchronizer.memory.update(currentAtomicMemory);
        deviceInfo() << "Updating memory";
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
    if (true == needUpdateConstant)
    {
        synchronizer.constant.update(currentAtomicConstant);
        deviceDebug() << "Updating constants";
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
    if (true == needUpdateMemory)
    {
        // Reset the stats, the memory was rebuilt
        miningStats.reset();
    }

    ////////////////////////////////////////////////////////////////////////////
    bool const resetStats{ false == config.occupancy.accumulateHash || true == needUpdateMemory };
    updateBatchNonce(resetStats);

    return true;
}


void device::Device::updateBatchNonce(bool const resetStats)
{
    ////////////////////////////////////////////////////////////////////////////
    common::Config const& config{ common::Config::instance() };

    ////////////////////////////////////////////////////////////////////////////
    uint32_t internalLoop{ 1u };
    if (std::nullopt != config.occupancy.internalLoop)
    {
        internalLoop = *config.occupancy.internalLoop;
    }

    ////////////////////////////////////////////////////////////////////////////
    miningStats.setBatchNonce(resolver->getBlocks() * resolver->getThreads() * internalLoop);

    ////////////////////////////////////////////////////////////////////////////
    if (true == resetStats)
    {
        miningStats.resetHashrate();
        miningStats.reset();
    }
}


void device::Device::loopDoWork()
{
    ////////////////////////////////////////////////////////////////////////////
    common::Config const& config{ common::Config::instance() };
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
    computing.store(true, boost::memory_order::seq_cst);

    ////////////////////////////////////////////////////////////////////////////
    miningStats.reset();

    ////////////////////////////////////////////////////////////////////////////
    deviceDebug() << "Start working!";
    while (true == isAlive() && nullptr != resolver)
    {
        // Check and update the job.
        // Do not compute directly after update device.
        // A new job can spawn during the update.
        if (true == updateJob())
        {
            continue;
        }

        // Execute the kernel to compute nonces.
        if (false == resolver->executeAsync(currentJobInfo))
        {
            kill(device::KILL_STATE::KERNEL_EXECUTE_FAIL);
            continue;
        }
        miningStats.increaseKernelExecuted();

        // Send share found
        submit(config.profile);

        // Increasing nonce to next kernel.
        currentJobInfo.nonce += miningStats.getBatchNonce();
    }

    ////////////////////////////////////////////////////////////////////////////
    cleanUp();

    ////////////////////////////////////////////////////////////////////////////
    computing.store(false, boost::memory_order::seq_cst);
}


void device::Device::submit(common::PROFILE const profile)
{
    ////////////////////////////////////////////////////////////////////////////
    switch (profile)
    {
        case common::PROFILE::STANDARD:
        {
            if (nullptr != resolver)
            {
                resolver->submit(stratum);
            }
            break;
        }
        case common::PROFILE::SMART_MINING:
        {
            if (nullptr != resolver)
            {
                resolver->submit(stratumSmartMining);
            }
            break;
        }
    }
}
