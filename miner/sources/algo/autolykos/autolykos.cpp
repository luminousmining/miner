#include <algo/autolykos/autolykos.hpp>
#include <common/log/log.hpp>


uint32_t algo::autolykos_v2::computePeriod(uint32_t const blockNumber)
{
    uint32_t period{ algo::autolykos_v2::EPOCH_MIN };

    if (blockNumber >= algo::autolykos_v2::BLOCK_END)
    {
        period = algo::autolykos_v2::EPOCH_MAX;
    }
    else
    {
        auto const begin{ algo::autolykos_v2::BLOCK_BEGIN };
        auto const epochPeriod{ algo::autolykos_v2::EPOCH_PERIOD };
        auto const blockDiff{ blockNumber - begin };
        auto const itersNumber{ (blockDiff / epochPeriod) + 1 };

        for (auto i{ 0u }; i < itersNumber; ++i)
        {
            period = period / 100u * 105u;
        }
    }

    return period;
}
