#pragma once

#include <vector>

#include <stratum/stratum.hpp>


namespace common
{
    namespace mocker
    {
        struct MockerStratum : public stratum::Stratum
        {
            uint32_t                        id{ 0u };
            boost::json::array              paramSubmit{};
            boost::json::object             paramSubmitObject{};
            std::vector<boost::json::array> allSubmits{};

            inline void onResponse(boost::json::object const&) final
            {
            }

            inline void miningSubmit(uint32_t const deviceID, boost::json::array const& params) final
            {
                id = deviceID;
                paramSubmit = params;
                allSubmits.push_back(params);
            }

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
