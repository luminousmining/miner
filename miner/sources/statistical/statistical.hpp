#pragma once


#include <common/chrono.hpp>


namespace statistical
{
    struct Statistical
    {
    public:
        void setChronoUnit(common::CHRONO_UNIT newUnit);
        void start();
        void stop();
        void reset();
        void increaseKernelExecuted();
        uint32_t getKernelExecutedCount() const;
        void setBatchNonce(uint64_t const newBatchNonce);
        uint64_t getBatchNonce() const;
        double getHashrate();

    private:
        common::CHRONO_UNIT chronoUnit { common::CHRONO_UNIT::US };
        common::Chrono chrono;
        uint64_t batchNonce { 0ull };
        uint32_t kernelExecuted { 0u };
    };
}