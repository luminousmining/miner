#include "statistical/statistical.hpp"

#include <chrono>
#include <thread>

#include <gtest/gtest.h>


using statistical::Statistical;


////////////////////////////////////////////////////////////////////////////////
// A window that completed real launches publishes a positive hashrate:
// rate = batchNonce * kernelExecuted / elapsed.
////////////////////////////////////////////////////////////////////////////////
TEST(StatisticalHashrate, WorkedWindowPublishesPositiveHashrate)
{
    Statistical stats{};
    stats.reset();

    stats.setBatchNonce(1000000ull);
    stats.increaseKernelExecuted();
    stats.increaseKernelExecuted();

    std::this_thread::sleep_for(std::chrono::milliseconds(2)); // guarantee elapsed > 0
    stats.stop();
    stats.updateHashrate();

    EXPECT_LT(0.0, stats.getHashrate());
}


////////////////////////////////////////////////////////////////////////////////
// Non-accumulate path: a job update resets the meter. A slow / memory-hard
// kernel that has not yet reached the publish threshold then reads 0 H/s, and
// the dashboard hides the device -- the bug --internal_accumulate_hash fixes.
////////////////////////////////////////////////////////////////////////////////
TEST(StatisticalHashrate, ResetHashrateZeroesDisplayedValue)
{
    Statistical stats{};
    stats.reset();

    stats.setBatchNonce(1000000ull);
    stats.increaseKernelExecuted();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    stats.stop();
    stats.updateHashrate();
    ASSERT_LT(0.0, stats.getHashrate());

    stats.resetHashrate();
    EXPECT_DOUBLE_EQ(0.0, stats.getHashrate());
    EXPECT_EQ(0u, stats.getKernelExecutedCount());
}


////////////////////////////////////////////////////////////////////////////////
// Accumulate path: a job update does NOT reset the meter, so the launch count
// and elapsed window carry across the boundary and the kernel keeps publishing
// a non-zero hashrate even when each job window is far shorter than the
// publish threshold.
////////////////////////////////////////////////////////////////////////////////
TEST(StatisticalHashrate, AccumulateAcrossJobUpdatePreservesHashrate)
{
    Statistical stats{};
    stats.reset();
    stats.setBatchNonce(1000000ull);

    // First job window.
    stats.increaseKernelExecuted();
    stats.increaseKernelExecuted();
    EXPECT_EQ(2u, stats.getKernelExecutedCount());

    // A new job arrives; in accumulate mode the meter is not reset, so the count
    // keeps growing rather than dropping back to zero.
    stats.increaseKernelExecuted();
    EXPECT_EQ(3u, stats.getKernelExecutedCount());

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    stats.stop();
    stats.updateHashrate();
    EXPECT_LT(0.0, stats.getHashrate());
}
