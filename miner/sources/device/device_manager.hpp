#pragma once


#include <vector>

#include <boost/thread/mutex.hpp>

#include <algo/algo_type.hpp>
#include <device/amd.hpp>
#include <device/nvidia.hpp>
#include <stratum/job_info.hpp>
#include <stratum/stratum.hpp>


namespace device
{
    class DeviceManager
    {
    public:
        DeviceManager();
        ~DeviceManager();

        static constexpr uint32_t WAITING_HASH_STATS{ 10000U };
        static constexpr uint32_t DEVICE_MAX_ID{ 1000000u };

        bool initialize();
        void run();
        void connectToPools();
        void onUpdateJob(uint32_t const _stratumUUID,
                         stratum::StratumJobInfo const& newJobInfo);
        void onShareStatus(bool const isValid,
                           uint32_t const requestID,
                           uint32_t const stratumUUID);

    private:
        uint32_t                 stratumUUID{ 0u };
        boost::thread            threadStatistical{};
        boost::mutex             mtxJobInfo{};
        std::vector<Device*>     devices{};
        stratum::StratumJobInfo  jobInfos[100];

        std::map<uint32_t/*DEVICE ID*/, stratum::Stratum*> stratums{};

        bool initializeStratum(uint32_t const deviceId,
                               algo::ALGORITHM const algorithm);
        bool initializeNvidia();
        bool initializeAmd();
        void updateDevice(uint32_t const _stratumUUID,
                          bool const updateMemory,
                          bool const updateConstants);
        bool containStratum(uint32_t const deviceId) const;
        stratum::Stratum* getOrCreateStratum(algo::ALGORITHM const algorithm,
                                             uint32_t const deviceId);
        void loopStatistical();
    };
}
