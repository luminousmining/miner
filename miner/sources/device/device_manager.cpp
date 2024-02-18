#include <string>

#include <CL/opencl.hpp>

#include <algo/algo_type.hpp>
#include <algo/hash_utils.hpp>
#include <common/cast.hpp>
#include <common/config.hpp>
#include <common/dashboard.hpp>
#include <common/custom.hpp>
#include <common/date.hpp>
#include <common/error/cuda_error.hpp>
#include <device/device_manager.hpp>
#include <stratum/autolykos_v2.hpp>
#include <stratum/etchash.hpp>
#include <stratum/ethash.hpp>
#include <stratum/evrprogpow.hpp>
#include <stratum/firopow.hpp>
#include <stratum/kawpow.hpp>
#include <stratum/sha256.hpp>


device::DeviceManager::DeviceManager()
{
}


device::DeviceManager::~DeviceManager()
{
    for ([[maybe_unused]] auto [_, stratum] : stratums)
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
    for (device::Device* device : devices)
    {
        if (nullptr != device)
        {
            std::optional<common::Config::PoolConfig> settings
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
                algo::ALGORITHM const customAlgo { config.getAlgorithm((*settings).algo) };
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

            logInfo() << "GPU[" << device->id << "] " << device->clDevice.getInfo<CL_DEVICE_BOARD_NAME_AMD>();
            devices.push_back(device);
        }
    }

    return true;
}


void device::DeviceManager::run()
{
    for (device::Device* const device : devices)
    {
        if (nullptr != device)
        {
            device->run();
        }
    }

    threadStatistical.interrupt();
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


void device::DeviceManager::loopStatistical()
{
    common::Dashboard board{};
    boost::chrono::milliseconds ms{ device::DeviceManager::WAITING_HASH_STATS };

    board.setTitle("HASHRATE");
    board.addColumn("DeviceID");
    board.addColumn("Coin");
    board.addColumn("Pool");
    board.addColumn("Hashrate");

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
            double const hashrate { device->getHashrate() };

            stratum::Stratum* stratum { nullptr };

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

            std::stringstream ss;
            ss << std::setprecision(4) << (hashrate / 1e6) << "MH/S";
            board.addLine
            (
                {
                    std::to_string(device->id),
                    algo::toString(device->algorithm),
                    stratum->host,
                    ss.str()
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
    uint32_t const _stratumUUID,
    stratum::StratumJobInfo const& newJobInfo)
{
    UNIQUE_LOCK(mtxJobInfo);

    stratum::StratumJobInfo& jobInfo { jobInfos[_stratumUUID] };

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
        updateDevice(_stratumUUID, updateMemory, updateConstants);
    }
}


void device::DeviceManager::updateDevice(
    uint32_t const _stratumUUID,
    bool const updateMemory,
    bool const updateConstants)
{
    for (device::Device* const device : devices)
    {
        if (   nullptr != device
            && device->getStratum()->uuid == _stratumUUID)
        {
            stratum::StratumJobInfo& jobInfo { jobInfos[_stratumUUID] };
            device->update(updateMemory, updateConstants, jobInfo);
        }
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

    switch (algorithm)
    {
        case algo::ALGORITHM::SHA256:
        {
            stratum = new (std::nothrow) stratum::StratumSha256;
            break;
        }
        case algo::ALGORITHM::ETHASH:
        {
            stratum = new (std::nothrow) stratum::StratumEthash;
            break;
        }
        case algo::ALGORITHM::ETCHASH:
        {
            stratum = new (std::nothrow) stratum::StratumEtchash;
            break;
        }
        case algo::ALGORITHM::PROGPOW:
        {
            stratum = new (std::nothrow) stratum::StratumProgPOW;
            break;
        }
        case algo::ALGORITHM::KAWPOW:
        {
            stratum = new (std::nothrow) stratum::StratumKawPOW;
            break;
        }
        case algo::ALGORITHM::FIROPOW:
        {
            stratum = new (std::nothrow) stratum::StratumFiroPOW;
            break;
        }
        case algo::ALGORITHM::EVRPROGPOW:
        {
            stratum = new (std::nothrow) stratum::StratumEvrprogPOW;
            break;
        }
        case algo::ALGORITHM::AUTOLYKOS_V2:
        {
            stratum = new (std::nothrow) stratum::StratumAutolykosV2;
            break;
        }
        case algo::ALGORITHM::UNKNOW:
        {
            break;
        }
    }

    stratum->uuid = stratumUUID++;
    if (stratumUUID >= 100u)
    {
        stratumUUID = 0u;
    }

    stratums[deviceId] = stratum;

    return stratum;
}
