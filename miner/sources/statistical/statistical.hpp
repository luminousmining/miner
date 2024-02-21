#pragma once


#include <common/chrono.hpp>


namespace statistical
{
    struct Statistical
    {
    public:
        struct alignas(64) ShareInfo
        {
            uint64_t total{ 0llu };
            uint64_t invalid{ 0llu };
            uint64_t valid{ 0llu };
        };

        void setChronoUnit(common::CHRONO_UNIT newUnit);
        void start();
        void stop();
        void reset();
        void increaseKernelExecuted();
        uint32_t getKernelExecutedCount() const;
        void setBatchNonce(uint64_t const newBatchNonce);
        uint64_t getBatchNonce() const;
        void updateHashrate();
        double getHahrate() const;
        ShareInfo& getShares();

    private:
        common::CHRONO_UNIT chronoUnit{ common::CHRONO_UNIT::US };
        common::Chrono      chrono;
        ShareInfo           shares{};
        uint64_t            batchNonce{ 0ull };
        double              hashrates{ 0.0 };
        uint32_t            kernelExecuted{ 0u };
    };
}