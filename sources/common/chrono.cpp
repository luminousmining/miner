#include <common/cast.hpp>
#include <common/chrono.hpp>
#include <common/log/log.hpp>


common::ChronoGuard::ChronoGuard(
    common::CHRONO_UNIT chronoUnit)
{
    unit = chronoUnit;
    tmStart = std::chrono::system_clock::now();
}


common::ChronoGuard::~ChronoGuard()
{
    std::chrono::system_clock::time_point const tmEnd { std::chrono::system_clock::now() };

    switch (unit)
    {
        case common::CHRONO_UNIT::SEC:
        {
            auto const elapsed { castSec(tmEnd - tmStart) };
            logInfo() << "elapsed: " << elapsed.count() << "s";
            break;
        }
        case common::CHRONO_UNIT::MS:
        {
            auto const elapsed { castMs(tmEnd - tmStart) };
            logInfo() << "elapsed: " << elapsed.count() << "ms";
            break;
        }
        case common::CHRONO_UNIT::US:
        {
            auto const elapsed { castUs(tmEnd - tmStart) };
            logInfo() << "elapsed: " << elapsed.count() << "us";
            break;
        }
        case common::CHRONO_UNIT::NS:
        {
            auto const elapsed { castNs(tmEnd - tmStart) };
            logInfo() << "elapsed: " << elapsed.count() << "ns";
            break;
        }
        default:
        {
            auto const elapsed { castMs(tmEnd - tmStart) };
            logInfo() << "elapsed: " << elapsed.count() << "ms";
            break;
        }
    }
}


void common::Chrono::start()
{
    tmStart = std::chrono::system_clock::now();
}


void common::Chrono::stop()
{
    tmEnd = std::chrono::system_clock::now();
}


uint64_t common::Chrono::elapsed(
    common::CHRONO_UNIT unit) const
{
    switch (unit)
    {
        case common::CHRONO_UNIT::SEC:
        {
            return castSec(tmEnd - tmStart).count();
        }
        case common::CHRONO_UNIT::MS:
        {
            return castMs(tmEnd - tmStart).count();
        }
        case common::CHRONO_UNIT::US:
        {
            return castUs(tmEnd - tmStart).count();
        }
        case common::CHRONO_UNIT::NS:
        {
            return castNs(tmEnd - tmStart).count();
        }
    }

    return castMs(tmEnd - tmStart).count();
}