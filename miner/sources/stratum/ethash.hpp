#pragma once


#include <algo/ethash/ethash.hpp>
#include <network/network.hpp>
#include <stratum/stratum.hpp>


namespace stratum
{
    struct StratumEthash : public stratum::Stratum
    {
    public:
        StratumEthash();
        virtual ~StratumEthash() = default;

        void onResponse(boost::json::object const& root) final;
        void onMiningNotify(boost::json::object const& root) final;
        void onMiningSetDifficulty(boost::json::object const& root) final;
        void onMiningSetTarget(boost::json::object const& root) final;
        void onMiningSetExtraNonce(boost::json::object const& root) final;

        void miningSubmit(uint32_t const deviceId,
                          boost::json::array const& params) final;

    protected:
        uint32_t maxEpochNumber { algo::ethash::MAX_EPOCH_NUMBER };
    };
}

