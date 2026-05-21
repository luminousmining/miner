#pragma once


#include <network/network.hpp>
#include <stratum/stratum.hpp>


namespace stratum
{
    class StratumRandomX : public stratum::Stratum
    {
      public:
        void onConnect() override;
        void onResponse(boost::json::object const& root) final;
        void onMiningNotify(boost::json::object const& root) final;
        void onMiningSetDifficulty(boost::json::object const& root) final;
        void onUnknownMethod(boost::json::object const& root) final;

        void miningSubmit(uint32_t const deviceId, boost::json::array const& params) final;

      private:
        std::string minerID{};

        void parseJob(boost::json::object const& job);
    };
}
