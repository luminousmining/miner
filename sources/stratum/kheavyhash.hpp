#pragma once

#include <network/network.hpp>
#include <stratum/stratum.hpp>


namespace stratum
{
    // Kaspa stratum (kaspa-stratum-bridge / gostratum dialect, NORMAL encoding).
    // mining.notify params = [ jobIdStr, [4 LE u64 pre_pow words], timestamp ].
    // mining.submit params = [ wallet.worker, jobIdStr, nonceHex ].
    class StratumKHeavyHash : public stratum::Stratum
    {
      public:
        void onResponse(boost::json::object const& root) final;
        void onMiningNotify(boost::json::object const& root) final;
        void onMiningSetDifficulty(boost::json::object const& root) final;
        void onMiningSetExtraNonce(boost::json::object const& root) final;
        void onUnknownMethod(boost::json::object const& root) final;

        void miningSubscribe() override;
        void miningSubmit(uint32_t const deviceId, boost::json::array const& params) final;
    };
}
