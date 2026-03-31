#pragma once


#include <network/network.hpp>
#include <stratum/stratum.hpp>


namespace stratum
{
    ////////////////////////////////////////////////////////////////////////////
    // Grin uses a custom JSON-RPC 2.0 stratum protocol distinct from the
    // Ethereum family.  Key differences from StratumProgPOW / StratumEthash:
    //
    //  Miner → Pool : "login"   (replaces mining.subscribe + mining.authorize)
    //  Pool → Miner : "job"     (replaces mining.notify)
    //  Miner → Pool : "submit"  (replaces mining.submit)
    //
    // The job payload contains:
    //   difficulty  – uint64 target difficulty
    //   height      – block height
    //   job_id      – uint32 pool-assigned identifier
    //   pre_pow     – hex-encoded block header prefix (≤ 384 bytes)
    //
    // The share payload contains:
    //   edge_bits   – always 32 for Cuckatoo32
    //   height      – same as job
    //   job_id      – same as job
    //   nonce       – uint64 found nonce
    //   pow         – array of 42 uint32 cycle nonces (sorted ascending)
    ////////////////////////////////////////////////////////////////////////////
    class StratumCuckatoo : public stratum::Stratum
    {
      public:
        void onResponse(boost::json::object const& root) final;
        void onMiningNotify(boost::json::object const& root) final;
        void onMiningSetDifficulty(boost::json::object const& root) final;
        void onUnknownMethod(boost::json::object const& root) final;

        void miningSubscribe() override;
        void miningSubmit(uint32_t const deviceId, boost::json::object const& params) final;

      private:
        void onJob(boost::json::object const& root);
        void miningLogin();
    };
}
