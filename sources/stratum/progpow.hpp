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
        void onResponse(boost::json::object const& root) override;
        void onMiningNotify(boost::json::object const& root) override;
        void onMiningSetDifficulty(boost::json::object const& root) final;
        void onMiningSetTarget(boost::json::object const& root) final;
        void onMiningSetExtraNonce(boost::json::object const& root) final;
        void onMiningSet(boost::json::object const& root) final;

        void miningSubmit(uint32_t const deviceId, boost::json::array const& params) override;

      protected:
        uint32_t maxPeriod{ algo::progpow::v_0_9_2::MAX_PERIOD };
        // Upper bound for the seedHash -> epoch search (a max epoch *number*, not a
        // per-epoch block count). Generous on purpose so the search covers every
        // realistic progpow/FiroPoW epoch; see deriveEpoch.
        uint32_t maxEpochSearch{ algo::ethash::EPOCH_LENGTH };
        // Blocks-per-epoch divisor for the blockNumber/length mapping.
        uint32_t maxEpochLength{ algo::progpow::EPOCH_LENGTH };

        // Derive the DAG epoch for the current job. The base maps the block number
        // onto EPOCH_LENGTH (what kawpow/meowpow/evrprogpow and progpow-quai expect);
        // FiroPoW overrides this to prefer the network seed hash. Returns -1 when no
        // epoch can be determined.
        virtual int32_t deriveEpoch(stratum::StratumJobInfo const& jobInfo) const;

      private:
        void onResponseEthereumV1(boost::json::object const& root);
        void onResponseEthereumV2(boost::json::object const& root);
        void onResponseEthProxy(boost::json::object const& root);
    };
}
