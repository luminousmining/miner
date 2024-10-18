#include <common/log/log.hpp>
#include <statistical/statistical.hpp>


void statistical::Statistical::setChronoUnit(
    common::CHRONO_UNIT newUnit)
{
    chronoUnit = newUnit;
}


void statistical::Statistical::start()
{
    chrono.start();
}


void statistical::Statistical::stop()
{
    chrono.stop();
}


void statistical::Statistical::reset()
{
    kernelExecuted = 0u;
    start();
}


void statistical::Statistical::increaseKernelExecuted()
{
    ++kernelExecuted;
}


uint32_t statistical::Statistical::getKernelExecutedCount() const
{
    return kernelExecuted;
}

void statistical::Statistical::setBatchNonce(
    uint64_t const newBatchNonce)
{
    batchNonce = newBatchNonce;
}


uint64_t statistical::Statistical::getBatchNonce() const
{
    return batchNonce;
}


void statistical::Statistical::updateHashrate()
{
    uint64_t elapsed { chrono.elapsed(chronoUnit) };
    double const diffInSecond { 1e6 / elapsed };
    uint64_t const totalNonce { batchNonce * kernelExecuted };
    double values { totalNonce * diffInSecond };

    if (values > 0.0)
    {
        hashrates = values;
    }
}


void statistical::Statistical::resetHashrate()
{
    kernelExecuted = 0u;
    hashrates = 0.0;
}


double statistical::Statistical::getHahrate() const
{
    return hashrates;
}


statistical::Statistical::ShareInfo& statistical::Statistical::getShares()
{
    return shares;
}
