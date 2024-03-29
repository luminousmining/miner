#include <string>

#include <CL/opencl.hpp>

#include <algo/algo_type.hpp>
#include <algo/hash_utils.hpp>
#include <common/cast.hpp>
#include <common/config.hpp>
#include <common/dashboard.hpp>
#include <common/custom.hpp>
#include <common/date.hpp>
#include <common/formater_hashrate.hpp>
#include <common/error/cuda_error.hpp>
#include <common/error/opencl_error.hpp>
#include <device/device_manager.hpp>
#include <stratum/stratums.hpp>


device::DeviceManager::~DeviceManager()
{
    for (auto [_, stratum] : stratums)
    {
        SAFE_DELETE(stratum);
    }
}


bool device::DeviceManager::initialize()
{
    common::Config const& config { common::Config::instance() };
    algo::ALGORITHM const algorithm { config.getAlgorithm() };

    ////////////////////////////////////////////////////////////////////////////
    if (true == config.deviceEnable.amdEnable)
    {
        if (false == initializeAmd())
        {
            logErr() << "Cannot initialize device Amd";
            return false;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == config.deviceEnable.nvidiaEnable)
    {
        if (false == initializeNvidia())
        {
            logErr() << "Cannot initialize device Nvidia";
            return false;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (common::PROFILE::SMART_MINING == config.profile)
    {
        initializeStratumSmartMining();
        for (device::Device* device : devices)
        {
            if (nullptr == device)
            {
                continue;
            }
            device->setStratumSmartMining(&stratumSmartMining);
        }
    }
    else
    {
        for (device::Device* device : devices)
        {
            if (nullptr == device)
            {
                continue;
            }

            std::optional < common::Config::PoolConfig > settings
            {
                config.getConfigDevice(device->id)
            };

            if (std::nullopt == settings)
            {
                if (false == initializeStratum(device::DeviceManager::DEVICE_MAX_ID, algorithm))
                {
                    return false;
                }
                stratum::Stratum* stratum { stratums.at(device::DeviceManager::DEVICE_MAX_ID) };
                device->setAlgorithm(algorithm);
                device->setStratum(stratum);
            }
            else
            {
                algo::ALGORITHM const customAlgo { algo::toEnum((*settings).algo) };
                if (false == initializeStratum(device->id, customAlgo))
                {
                    return false;
                }
                stratum::Stratum* stratum { stratums.at(device->id) };
                device->setAlgorithm(customAlgo);
                device->setStratum(stratum);
            }
        }
    }

    return true;
}

bool device::DeviceManager::initializeStratumSmartMining()
{
    common::Config const& config { common::Config::instance() };

    stratumSmartMining.host.assign("192.168.1.25");
    stratumSmartMining.port = 8080;
    stratumSmartMining.workerName.assign(config.mining.workerName);
    stratumSmartMining.password.assign(config.mining.password);

    stratumSmartMining.setCallbackSetAlgorithm(
        std::bind(
            &device::DeviceManager::onSmartMiningSetAlgorithm,
            this,
            std::placeholders::_1));

    stratumSmartMining.setCallbackUpdateJob(
        std::bind(
            &device::DeviceManager::onSmartMiningUpdateJob,
            this,
            std::placeholders::_1));

    stratumSmartMining.setCallbackShareStatus(
        std::bind(
            &device::DeviceManager::onShareStatus,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3));

    return true;
}


bool device::DeviceManager::initializeStratum(
    uint32_t const deviceId,
    algo::ALGORITHM const algorithm)
{
    stratum::Stratum* stratum { nullptr };
    common::Config const& config { common::Config::instance() };

    if (true == containStratum(deviceId))
    {
        return true;
    }

    stratum = getOrCreateStratum(algorithm, deviceId);
    if (nullptr == stratum)
    {
        logErr() << "Cannot alloc memory for stratum!";
        return false;
    }

    stratum->algorithm = algorithm;

    if (deviceId == device::DeviceManager::DEVICE_MAX_ID)
    {
        stratum->host.assign(config.mining.host);
        stratum->port = config.mining.port;
        stratum->workerName.assign(config.mining.workerName);
        stratum->wallet.assign(config.mining.wallet);
        stratum->password.assign(config.mining.password);
    }
    else
    {
        std::optional<common::Config::PoolConfig> customSettings
        {
            config.getConfigDevice(deviceId)
        };
        if (std::nullopt == customSettings)
        {
            logErr() << "Device have not custom settings";
            return false;
        }
        stratum->host.assign((*customSettings).host);
        stratum->port = (*customSettings).port;
        stratum->workerName.assign((*customSettings).workerName);
        stratum->wallet.assign((*customSettings).wallet);
        stratum->password.assign((*customSettings).password);
    }

    stratum->setCallbackUpdateJob(
        std::bind(
            &device::DeviceManager::onUpdateJob,
            this,
            std::placeholders::_1,
            std::placeholders::_2));

    stratum->setCallbackShareStatus(
        std::bind(
            &device::DeviceManager::onShareStatus,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3));

    return true;
}


bool device::DeviceManager::initializeNvidia()
{
    int32_t numberDevice{ 0 };
    CUDA_ER(cudaGetDeviceCount(&numberDevice));

    if (0u == numberDevice)
    {
        return true;
    }

    for (int32_t i{ 0 }; i < numberDevice; ++i)
    {
        device::DeviceNvidia* device{ new device::DeviceNvidia };
        device->deviceType = device::DEVICE_TYPE::NVIDIA;
        cudaError_t const codeError{ cudaGetDeviceProperties(&device->properties, i) };
        if (cudaSuccess != codeError)
        {
            logErr() << "[" << codeError << "]" << __FUNCTION__ << cudaGetErrorString(codeError);
            delete device;
            device = nullptr;
            continue;
        }
        device->cuIndex = castU32(i);
        device->id = castU32(devices.size());
        device->pciBus = device->properties.pciBusID;

        logInfo() << "GPU[" << devices.size() << "] " << device->properties.name;
        devices.push_back(device);
    }

    return true;
}


bool device::DeviceManager::initializeAmd()
{
    std::vector<cl::Device> cldevices{};
    std::vector<cl::Platform> platforms{};
    cl::Platform::get(&platforms);

    // Get all OpenCL devices
    // GPU AMD
    for (cl::Platform const& platform : platforms)
    {
        std::string const platformName { platform.getInfo<CL_PLATFORM_NAME>() };
        if (platformName.find("AMD") == std::string::npos)
        {
            continue;
        }

        platform.getDevices(CL_DEVICE_TYPE_GPU, &cldevices);
        for (uint32_t i { 0u }; i < cldevices.size(); ++i)
        {
            device::DeviceAmd* device{ new device::DeviceAmd };
            device->deviceType = device::DEVICE_TYPE::AMD;

            device->clDevice = cldevices.at(i);
            if (CL_DEVICE_TYPE_GPU != device->clDevice.getInfo<CL_DEVICE_TYPE>())
            {
                SAFE_DELETE(device);
                continue;
            }
            device->id = castU32(devices.size());

            cl_char topology[24]{ 0, };
            OPENCL_ER(
                clGetDeviceInfo(
                    device->clDevice.get(),
                    CL_DEVICE_TOPOLOGY_AMD,
                    sizeof(topology),
                    &topology,
                    nullptr));
            device->pciBus = castU32(topology[21]);

            logInfo() << "GPU[" << device->id << "] " << device->clDevice.getInfo<CL_DEVICE_BOARD_NAME_AMD>();
            devices.push_back(device);
        }
    }

    return true;
}


void device::DeviceManager::run()
{
    ////////////////////////////////////////////////////////////////////////////
    stratum::StratumJobInfo cleanJob{};

    ////////////////////////////////////////////////////////////////////////////
    threadStatistical.interrupt();

    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i { 0u }; i < device::DeviceManager::MAX_STRATUMS; ++i)
    {
        stratum::StratumJobInfo& jobInfo { jobInfos[i] };
        jobInfo = cleanJob;
    }

    ////////////////////////////////////////////////////////////////////////////
    for (device::Device* const device : devices)
    {
        if (nullptr != device)
        {
            device->cleanJob();
            device->run();
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    threadStatistical = boost::thread{ boost::bind(&device::DeviceManager::loopStatistical, this) };
}


void device::DeviceManager::connectToPools()
{
    for (auto [_, stratum] : stratums)
    {
        if (false == stratum->connect())
        {
            return;
        }
    }

    for (auto [_, stratum] : stratums)
    {
        stratum->wait();
    }
}


void device::DeviceManager::connectToSmartMining()
{
    if (true == stratumSmartMining.connect())
    {
        stratumSmartMining.wait();
    }
}


std::vector<device::Device*>& device::DeviceManager::getDevices()
{
    return devices;
}


void device::DeviceManager::loopStatistical()
{
    std::string host{};
    common::Dashboard board{};
    stratum::Stratum* stratum { nullptr };
    common::Config const& config { common::Config::instance() };
    boost::chrono::milliseconds ms{ device::DeviceManager::WAITING_HASH_STATS };

    board.setTitle("HASHRATE");
    board.addColumn("DeviceID");
    board.addColumn("Algorithm");
    board.addColumn("Pool");
    board.addColumn("Hashrate");
    board.addColumn("Shares");

    while (true)
    {
        ////////////////////////////////////////////////////////////////////////
        boost::this_thread::sleep_for(ms);

        ////////////////////////////////////////////////////////////////////////
        board.resetLines();
        board.setDate(common::getDate());

        ////////////////////////////////////////////////////////////////////////
        bool displayable{ false };
        for (device::Device* const device : devices)
        {
            if (   nullptr == device
                || false == device->isAlive())
            {
                continue;
            }
 
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

            double const hashrate { device->getHashrate() };
            statistical::Statistical::ShareInfo shares { device->getShare() };

            std::stringstream ssShares;
            ssShares << shares.valid << "/" << shares.invalid << "/" << shares.total;

            board.addLine
            (
                {
                    std::to_string(device->id),
                    algo::toString(device->algorithm),
                    host,
                    common::hashrateToString(hashrate),
                    ssShares.str()
                }
            );

            if (hashrate > 0.0)
            {
                displayable = true;
            }
        }

        ////////////////////////////////////////////////////////////////////////
        if (true == displayable)
        {
            board.show();
        }
    }
}


void device::DeviceManager::onUpdateJob(
    uint32_t const stratumUUID,
    stratum::StratumJobInfo const& newJobInfo)
{
    UNIQUE_LOCK(mtxJobInfo);

    stratum::StratumJobInfo& jobInfo { jobInfos[stratumUUID] };

    bool updateMemory { false };
    bool updateConstants { false };
    bool const isSameEpoch { jobInfo.epoch == newJobInfo.epoch };
    bool const isSameHeader { algo::isEqual(jobInfo.headerHash, newJobInfo.headerHash) };

    if (true == isSameHeader && true == isSameEpoch)
    {
#if defined(_DEBUG)
        logDebug()
            << "Skip Job"
            << ", isSameEpoch " << std::boolalpha << isSameEpoch << std::dec
            << ", isSameHeader " << std::boolalpha << isSameHeader << std::dec
            << newJobInfo;
#endif
        return;
    }

    if (false == isSameEpoch)
    {
        updateMemory = true;
    }

    if (false == isSameHeader)
    {
        updateConstants = true;
    }

    if (true == updateMemory || true == updateConstants)
    {
        jobInfo = newJobInfo;
        jobInfo.gapNonce /= devices.size();
#if defined(_DEBUG)
        logInfo() << jobInfo;
#else
        logInfo() << "New Job[" << jobInfo.jobID << "]";
#endif
        updateDevice(stratumUUID, updateMemory, updateConstants);
    }
}


void device::DeviceManager::onShareStatus(
    bool const isValid,
    uint32_t const requestID,
    uint32_t const stratumUUID)
{
    common::Config const& config { common::Config::instance() };

    for (device::Device* const device : devices)
    {
        if (nullptr == device)
        {
            continue;
        }
        if (common::PROFILE::STANDARD == config.profile)
        {
            stratum::Stratum* stratum { device->getStratum() };
            if (   nullptr == stratum
                || stratum->uuid != stratumUUID)
            {
                continue;
            }
        }

        uint32_t const shareID { (device->id + 1u) * stratum::Stratum::OVERCOM_NONCE };
        if (shareID == requestID)
        {
            device->increaseShare(isValid);
            return;
        }
    }
}


void device::DeviceManager::onSmartMiningSetAlgorithm(
    algo::ALGORITHM const algorithm)
{
    threadStatistical.interrupt();

    boost::chrono::milliseconds ms{ device::DeviceManager::WAITING_DEVICE_STOP_COMPUTE };

    for (device::Device* device : devices)
    {
        if (nullptr == device)
        {
            continue;
        }

        device->kill(device::KILL_STATE::DISABLE);
        while (true == device->isComputing())
        {
            boost::this_thread::sleep_for(ms);
        }
    }

    for (device::Device* device : devices)
    {
        if (nullptr == device)
        {
            continue;
        }

        device->setStratumSmartMining(&stratumSmartMining);
        device->setAlgorithm(algorithm);
    }

    run();
}


void device::DeviceManager::onSmartMiningUpdateJob(
    stratum::StratumJobInfo const& newJobInfo)
{
    onUpdateJob(0u, newJobInfo);
}


void device::DeviceManager::updateDevice(
    uint32_t const stratumUUID,
    bool const updateMemory,
    bool const updateConstants)
{
    common::Config const& config { common::Config::instance() };

    for (device::Device* const device : devices)
    {
        if (nullptr == device)
        {
            continue;
        }

        if (common::PROFILE::STANDARD == config.profile)
        {
            stratum::Stratum* stratum { device->getStratum() };
            if (   nullptr == stratum
                || stratum->uuid != stratumUUID)
            {
                continue;
            }
        }

        stratum::StratumJobInfo& jobInfo { jobInfos[stratumUUID] };
        device->update(updateMemory, updateConstants, jobInfo);
    }
}


bool device::DeviceManager::containStratum(
    uint32_t const deviceId) const
{
    return stratums.find(deviceId) != stratums.end();
}


stratum::Stratum* device::DeviceManager::getOrCreateStratum(
    algo::ALGORITHM const algorithm,
    uint32_t const deviceId)
{
    stratum::Stratum* stratum { nullptr };

    auto it { stratums.find(deviceId) };
    if (it != stratums.end())
    {
        return it->second;
    }

    stratum = stratum::NewStratum(algorithm);

    if (nullptr == stratum)
    {
        return nullptr;
    }

    stratum->uuid = stratumCount++;
    if (stratumCount >= 100u)
    {
        stratumCount = 0u;
    }

    stratums[deviceId] = stratum;

    return stratum;
}
