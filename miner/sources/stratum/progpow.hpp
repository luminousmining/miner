#pragma once


#include <algo/ethash/ethash.hpp>
#include <algo/progpow/progpow.hpp>
#include <network/network.hpp>
#include <stratum/stratum.hpp>


namespace stratum
{
    struct StratumProgPOW : public stratum::Stratum
    {
    public:
        StratumProgPOW();
        virtual ~StratumProgPOW() = default;

        void onResponse(boost::json::object const& root) final;
        void onMiningNotify(boost::json::object const& root) final;
        void onMiningSetDifficulty(boost::json::object const& root) final;
        void onMiningSetTarget(boost::json::object const& root) final;
        void onMiningSetExtraNonce(boost::json::object const& root) final;

        void miningSubmit(uint32_t const deviceId,
                          boost::json::array const& params) final;

    protected:
        uint32_t maxPeriod { algo::progpow::v_0_9_2::MAX_PERIOD };
        uint32_t maxEthashEpoch { algo::ethash::EPOCH_LENGTH };
        uint32_t maxEpochLength { algo::progpow::EPOCH_LENGTH };
    };
}

