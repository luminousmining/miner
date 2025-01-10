#include <algo/ethash/ethash.hpp>
#include <algo/progpow/progpow_quai.hpp>
#include <algo/hash_utils.hpp>
#include <common/boost_utils.hpp>
#include <common/custom.hpp>
#include <stratum/progpow_quai.hpp>


stratum::StratumProgpowQuai::StratumProgpowQuai() :
    stratum::StratumProgPOW()
{
    stratumType = stratum::STRATUM_TYPE::ETHEREUM_V2;
    stratumName = "EthereumStratum/2.0.0";
}


void stratum::StratumProgpowQuai::onResponse(
    boost::json::object const& root)
{
    using namespace std::string_literals;

    auto const miningRequestID{ common::boostJsonGetNumber<uint32_t>(root.at("id")) };
    auto const requestID{ static_cast<stratum::ETHEREUM_V2_ID>(miningRequestID) };

    switch(requestID)
    {
        case stratum::ETHEREUM_V2_ID::MINING_HELLO:
        {
            if (   false == root.contains("error")
                || true == root.at("error").is_null())
            {
                auto const& result{ root.at("result").as_object() };

                std::string const encoding{ common::boostGetString(result, "encoding"s) };
                std::string const node{ common::boostGetString(result, "node"s) };
                std::string const proto{ common::boostGetString(result, "proto"s) };
                uint32_t const maxerrors{ common::boostJsonGetNumber<uint32_t>(result, "maxerrors"s) };
                uint32_t const resume{ common::boostJsonGetNumber<uint32_t>(result, "resume"s) };
                uint32_t const timeout{ common::boostJsonGetNumber<uint32_t>(result, "timeout"s) };

                if (   "plain"s == encoding
                    && false == node.empty()
                    && false == proto.empty()
                    && 0u < maxerrors)
                {
                    miningSubscribe();
                }
            }
            else
            {
                logErr() << "Hello failed: " << root;
            }
            break;
        }
        case stratum::ETHEREUM_V2_ID::MINING_SUBSCRIBE:
        {
            if (   false == root.contains("error")
                || true == root.at("error").is_null())
            {
                miningAuthorize();
            }
            else
            {
                logErr() << "Subscribe failed: " << root;
            }

            break;
        }
        case stratum::ETHEREUM_V2_ID::MINING_AUTHORIZE:
        {
            break;
        }
        default:
        {
            break;
        }
    }
}


void stratum::StratumProgpowQuai::onMiningSet(
    boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    boost::json::object const& params(root.at("params").as_object());

    ////////////////////////////////////////////////////////////////////////////
    std::string const algo{ common::boostGetString(params, "algo") };
    std::string const epoch{ common::boostGetString(params, "epoch") };
    std::string const extraNonce{ common::boostGetString(params, "extranonce") };
    std::string const target{ common::boostGetString(params, "target") };

    setExtraNonce(extraNonce);

    jobInfo.epoch = std::stoull(epoch);
    jobInfo.boundary = algo::toHash256(target);
    jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
    if (jobInfo.boundaryU64 < jobInfo.targetBits)
    {
        jobInfo.boundaryU64 = jobInfo.targetBits;
    }
}


void stratum::StratumProgpowQuai::onMiningNotify(
    boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    boost::json::array const& params(root.at("params").as_array());

    ////////////////////////////////////////////////////////////////////////////
    std::string jobID{ params.at(0).as_string().c_str() };
    std::string blockHeight{ params.at(1).as_string().c_str() };
    std::string header{ params.at(2).as_string().c_str() };

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.headerHash = algo::toHash256(header);
    jobInfo.jobID = algo::toHash256(jobID);
    jobInfo.jobIDStr.assign(jobID);
}
