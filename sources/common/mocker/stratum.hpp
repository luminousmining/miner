#pragma once

#include <stratum/stratum.hpp>


namespace common
{
    namespace mocker
    {
        struct MockerStratum : public stratum::Stratum
        {
            uint32_t            id{ 0u };
            boost::json::array  paramSubmit{};
            boost::json::object paramSubmitObject{};

            inline void onResponse(boost::json::object const&) final
            {
            }

            inline void miningSubmit(uint32_t const deviceID, boost::json::array const& params) final
            {
                id = deviceID;
                paramSubmit = params;
            }

            // Object-form submit (e.g. Blake3/Alephium): the base logs "not implemented",
            // so capture it here too for tests that submit object params.
            inline void miningSubmit(uint32_t const deviceID, boost::json::object const& params) final
            {
                id = deviceID;
                paramSubmitObject = params;
            }

            inline void onMiningNotify(boost::json::object const&) final
            {
            }

            inline void onMiningSetDifficulty(boost::json::object const&) final
            {
            }

            inline void onMiningSetTarget(boost::json::object const&) final
            {
            }

            inline void onMiningSetExtraNonce(boost::json::object const&) final
            {
            }
        };
    }
}
