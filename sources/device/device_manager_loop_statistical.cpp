#include <string>


#include <algo/algo_type.hpp>
#include <common/config.hpp>
#include <common/date.hpp>
#include <common/formater_hashrate.hpp>
#include <common/number_to_string.hpp>
#include <device/device_manager.hpp>
#include <stratum/stratums.hpp>


void device::DeviceManager::loopStatistical()
{
    std::string host{};
    common::Dashboard boardMining{};
    common::Dashboard boardDevice{};
    stratum::Stratum* stratum{ nullptr };
    common::Config const& config{ common::Config::instance() };
    boost::chrono::milliseconds ms{ device::DeviceManager::WAITING_HASH_STATS };

    ////////////////////////////////////////////////////////////////////////
    boardMining.setTitle("HASHRATE");
    boardMining.addColumn("Type");
    boardMining.addColumn("ID");
    boardMining.addColumn("Pci");
    boardMining.addColumn("Algorithm");
    boardMining.addColumn("Pool");
    boardMining.addColumn("Hashrate");
    boardMining.addColumn("Valid");
    boardMining.addColumn("Reject");

    ////////////////////////////////////////////////////////////////////////
    boardDevice.setTitle("USAGE");
    boardDevice.addColumn("Type");
    boardDevice.addColumn("ID");
    boardDevice.addColumn("Pci");
    boardDevice.addColumn("Power");
    boardDevice.addColumn("CoreClock");
    boardDevice.addColumn("MemoryClock");
    boardDevice.addColumn("Utilization");
    boardDevice.addColumn("H/W");

    while (true)
    {
        ////////////////////////////////////////////////////////////////////////
        boost::this_thread::sleep_for(ms);

        ////////////////////////////////////////////////////////////////////////
        boardMining.resetLines();
        boardMining.setDate(common::getDate());

        ////////////////////////////////////////////////////////////////////////
        boardDevice.resetLines();
        boardDevice.setDate(common::getDate());

        ////////////////////////////////////////////////////////////////////////
        bool displayable{ false };
        for (device::Device* const device : devices)
        {
            ///////////////////////////////////////////////////////////////////
            if (   nullptr == device
                || false == device->isAlive())
            {
                continue;
            }
 
            ///////////////////////////////////////////////////////////////////
            if (common::PROFILE::STANDARD == config.profile)
            {
                auto const& itStratum { stratums.find(device->id) };
                if (itStratum != stratums.end())
                {
                    stratum = itStratum->second;
                }
                else
                {
                    stratum = stratums.at(device::DeviceManager::DEVICE_MAX_ID);
                }

                if (nullptr == stratum)
                {
                    continue;
                }
                host.assign(stratum->host);
            }
            else
            {
                host.assign("smart_mining");
            }

            ///////////////////////////////////////////////////////////////////
            auto const hashrate{ device->getHashrate() };
            statistical::Statistical::ShareInfo shareInfo{ device->getShare() };

            ///////////////////////////////////////////////////////////////////
            showMiningStats(boardMining, device, hashrate, host, shareInfo);
            showDeviceStats(boardDevice, device, hashrate);

            ///////////////////////////////////////////////////////////////////
            if (0.0 < hashrate)
            {
                displayable = true;
            }
        }

        ////////////////////////////////////////////////////////////////////////
        if (true == displayable)
        {
            boardMining.show();
            boardDevice.show();
        }
    }
}


void device::DeviceManager::showMiningStats(
    common::Dashboard& board,
    device::Device* const device,
    double const hashrate,
    std::string const& host,
    statistical::Statistical::ShareInfo const& shareInfo)
{
    ///////////////////////////////////////////////////////////////////
    std::string deviceType{ "UNKNOW" };
    switch(device->deviceType)
    {
#if defined(CUDA_ENABLE)
        case device::DEVICE_TYPE::NVIDIA:
        {
            deviceType = "NVIDIA";
            break;
        }
#endif
#if defined(AMD_ENABLE)
        case device::DEVICE_TYPE::AMD:
        {
            deviceType = "AMD";
            break;
        }
#endif
        case device::DEVICE_TYPE::UNKNOW:
        {
            deviceType = "UNKNOW";
            break;
        }
    }

    ///////////////////////////////////////////////////////////////////
    board.addLine
    (
        {
            deviceType,
            std::to_string(device->id),
            std::to_string(device->pciBus),
            algo::toString(device->algorithm),
            host,
            common::hashrateToString(hashrate),
            std::to_string(shareInfo.valid),
            std::to_string(shareInfo.invalid)
        }
    );
}


void device::DeviceManager::showDeviceStats(
    common::Dashboard& board,
    device::Device* const device,
    double const hashrate)
{
    ///////////////////////////////////////////////////////////////////
    double power{ 0.0 };
    double hashByPower{ 0.0 };
    uint32_t coreClock{ 0u };
    uint32_t memoryClock{ 0u };
    uint32_t utilizationPercent{ 0u };

    ///////////////////////////////////////////////////////////////////
    std::string deviceType{ "UNKNOW" };
    switch(device->deviceType)
    {
#if defined(CUDA_ENABLE)
        case device::DEVICE_TYPE::NVIDIA:
        {
            deviceType = "NVIDIA";
            break;
        }
#endif
#if defined(AMD_ENABLE)
        case device::DEVICE_TYPE::AMD:
        {
            deviceType = "AMD";
            break;
        }
#endif
        case device::DEVICE_TYPE::UNKNOW:
        {
            deviceType = "UNKNOW";
            break;
        }
    }

    ///////////////////////////////////////////////////////////////////
    switch(device->deviceType)
    {
#if defined(CUDA_ENABLE)
        case device::DEVICE_TYPE::NVIDIA:
        {
            if (   nullptr != device->deviceNvml
                && true == profilerNvidia.valid)
            {
                power = profilerNvidia.getPowerUsage(device->deviceNvml);
                coreClock = profilerNvidia.getCoreClock(device->deviceNvml);
                memoryClock = profilerNvidia.getMemoryClock(device->deviceNvml);
                utilizationPercent = profilerNvidia.getUtilizationRate(device->deviceNvml);
            }
            break;
        }
#endif
#if defined(AMD_ENABLE)
        case device::DEVICE_TYPE::AMD:
        {
            if (true == profilerAmd.valid)
            {
                auto const activity{ profilerAmd.getCurrentActivity(device->id) };
                coreClock = activity.iEngineClock;
                memoryClock = activity.iMemoryClock;
                utilizationPercent = activity.iActivityPercent;
            }
            break;
        }
#endif
        case device::DEVICE_TYPE::UNKNOW:
        {
            break;
        }
    }

    if (0.0 < power)
    {
        hashByPower = hashrate / power;
    }

    ///////////////////////////////////////////////////////////////////
    board.addLine
    (
        {
            deviceType,
            std::to_string(device->id),
            std::to_string(device->pciBus),
            common::doubleToString(power),
            std::to_string(coreClock),
            std::to_string(memoryClock),
            std::to_string(utilizationPercent),
            common::hashrateToString(hashByPower)
        }
    );
}
