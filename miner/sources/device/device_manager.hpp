#pragma once


#include <vector>

#include <boost/thread/mutex.hpp>

#include <algo/algo_type.hpp>
#include <device/amd.hpp>
#include <device/nvidia.hpp>
#include <stratum/job_info.hpp>
#include <stratum/stratum.hpp>
#include <stratum/smart_mining.hpp>


namespace device
{
    class DeviceManager
    {
    public:
        DeviceManager() = default;
        ~DeviceManager();

        static constexpr uint32_t MAX_STRATUMS { 100u };
        static constexpr uint32_t WAITING_DEVICE_STOP_COMPUTE{ 100u };
        static constexpr uint32_t WAITING_HASH_STATS{ 10000u };
        static constexpr uint32_t DEVICE_MAX_ID{ 1000000u };

        bool initialize();
        void run();
        void connectToPools();
        void connectToSmartMining();
        std::vector<Device*>& getDevices();

        void onUpdateJob(uint32_t const stratumUUID,
                         stratum::StratumJobInfo const& newJobInfo);
        void onShareStatus(bool const isValid,
                           uint32_t const requestID,
                           uint32_t const stratumUUID);

        void onSmartMiningSetAlgorithm(algo::ALGORITHM const algorithm);
        void onSmartMiningUpdateJob(stratum::StratumJobInfo const& newJobInfo);

    private:
        uint32_t                 stratumCount{ 0u };
        boost::thread            threadStatistical{};
        boost::mutex             mtxJobInfo{};
        std::vector<Device*>     devices{};
        stratum::StratumJobInfo  jobInfos[MAX_STRATUMS];

        stratum::StratumSmartMining                        stratumSmartMining{};
        std::map<uint32_t/*DEVICE ID*/, stratum::Stratum*> stratums{};

        bool initializeStratum(uint32_t const deviceId,
                               algo::ALGORITHM const algorithm);
        bool initializeStratumSmartMining();
#if defined(CUDA_ENABLE)
        bool initializeNvidia();
#endif
#if defined(AMD_ENABLE)
        bool initializeAmd();
#endif
        void updateDevice(uint32_t const stratumUUID,
                          bool const updateMemory,
                          bool const updateConstants);
        bool containStratum(uint32_t const deviceId) const;
        stratum::Stratum* getOrCreateStratum(algo::ALGORITHM const algorithm,
                                             uint32_t const deviceId);
        void loopStatistical();
    };
}
