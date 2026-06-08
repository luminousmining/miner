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
        uint32_t maxEthashEpoch{ algo::ethash::EPOCH_LENGTH };
        uint32_t maxEpochLength{ algo::progpow::EPOCH_LENGTH };

        // Quai decouples the network seed hash from the DAG-size epoch: its pool
        // seed hash is keccak-iterated far more than blockNumber/EPOCH_LENGTH, so
        // matching the seed (findEpoch) yields a hugely inflated epoch and an
        // un-allocatable multi-tens-of-GiB DAG. The official quai-gpu-miner derives
        // the epoch from blockNumber/EPOCH_LENGTH and ignores the seed hash. Set
        // this for such chains; leave false to keep the seed-first derivation that
        // FiroPoW relies on (its EPOCH_LENGTH changed across the chain's history).
        bool deriveEpochFromBlockNumber{ false };

      private:
        void onResponseEthereumV1(boost::json::object const& root);
        void onResponseEthereumV2(boost::json::object const& root);
        void onResponseEthProxy(boost::json::object const& root);
    };
}
