#include <common/log/log.hpp>
#include <common/cast.hpp>
#include <statistical/statistical.hpp>


void statistical::Statistical::setChronoUnit(
    common::CHRONO_UNIT newUnit)
{
    chronoUnit = newUnit;
    switch(chronoUnit)
    {
        case common::CHRONO_UNIT::SEC:
        {
            chronoTime = 1;
            break;
        }
        case common::CHRONO_UNIT::MS:
        {
            chronoTime = castDouble(common::SEC_TO_MS);
            break;
        }
        case common::CHRONO_UNIT::US:
        {
            chronoTime = castDouble(common::SEC_TO_US);
            break;
        }
        case common::CHRONO_UNIT::NS:
        {
            chronoTime = castDouble(common::SEC_TO_NS);
            break;
        }
    }
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
    uint64_t const elapsed{ chrono.elapsed(chronoUnit) };
    double const diffTime{ chronoTime / elapsed };
    uint64_t const totalNonce{ batchNonce * kernelExecuted };
    double const values{ totalNonce * diffTime };

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


double statistical::Statistical::getHashrate() const
{
    return hashrates;
}


statistical::Statistical::ShareInfo& statistical::Statistical::getShares()
{
    return shares;
}
