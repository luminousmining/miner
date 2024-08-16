#pragma once

#include <cstdint>

#include <boost/thread.hpp>
#include <boost/atomic/atomic.hpp>
#include <boost/thread/mutex.hpp>

#include <algo/algo_type.hpp>
#include <device/type.hpp>
#include <network/network.hpp>
#include <statistical/statistical.hpp>
#include <stratum/smart_mining.hpp>
#include <stratum/stratum.hpp>
#include <resolver/resolver.hpp>


namespace device
{
    enum class KILL_STATE : uint32_t
    {
        ALGORITH_UNDEFINED,
        RESOLVER_NULLPTR,
        UPDATE_MEMORY_FAIL,
        UPDATE_CONSTANT_FAIL,
        KERNEL_EXECUTE_FAIL,
        DISABLE
    };

    struct Device
    {
    public:
        uint32_t            id{ 0u };
        uint32_t            pciBus { 0u };
        device::DEVICE_TYPE deviceType { device::DEVICE_TYPE::UNKNOW };
        algo::ALGORITHM     algorithm { algo::ALGORITHM::UNKNOW };
        uint32_t            stratumUUID { 0u };

        Device() = default;
        virtual ~Device() = default;

        Device(Device const&) = delete;
        Device(Device&) = delete;
        Device& operator=(Device const&) = delete;
        Device& operator=(Device&&) = delete;

        void run();
        void setStratum(stratum::Stratum* const newStratum);
        void setStratumSmartMining(stratum::StratumSmartMining* const newStratum);
        void setAlgorithm(algo::ALGORITHM newAlgorithm);
        void cleanJob();
        void kill(device::KILL_STATE const state);
        bool isAlive() const;
        bool isComputing() const;
        void update(bool const memory,
                    bool const constants,
                    stratum::StratumJobInfo const& newJobInfo);
        void increaseShare(bool const isValid);
        double getHashrate();
        stratum::Stratum* getStratum();
        stratum::StratumSmartMining* getStratumSmartMining();
        statistical::Statistical::ShareInfo getShare();

    protected:
        statistical::Statistical miningStats{};
        resolver::Resolver*      resolver{ nullptr };

        virtual bool initialize() = 0;
        virtual void cleanUp() = 0;

        bool updateJob();
        void waitJob();
        void loopDoWork();

    private:
        uint32_t                     kernelMinimunExecuteNeeded { 500u };
        boost::atomic_bool           alive { false };
        boost::atomic_bool           computing { false };
        boost::thread                threadDoWork{};
        boost::mutex                 mtxUpdate{};
        boost::condition_variable    notifyNewWork{};
        boost::atomic_bool           needUpdateMemory{ false };
        boost::atomic_bool           needUpdateConstants{ false };
        stratum::Stratum*            stratum{ nullptr };
        stratum::StratumSmartMining* stratumSmartMining{ nullptr };
        stratum::StratumJobInfo      nextjobInfo{};
        stratum::StratumJobInfo      currentJobInfo{};
    };
}
