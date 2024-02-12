#include <string>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/autolykos/autolykos.hpp>
#include <algo/autolykos/cuda/autolykos.cuh>
#include <common/boost_utils.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/autolykos_v2.hpp>


void stratum::StratumAutolykosV2::onResponse(
    boost::json::object const& root)
{
    auto const miningRequestID{ common::boostJsonGetNumber<uint32_t>(root.at("id")) };

    switch(miningRequestID)
    {
        case stratum::Stratum::ID_MINING_SUBSCRIBE:
        {
            if (true == root.at("error").is_null())
            {
                auto result{ root.at("result").as_array() };
                if (false == result.empty())
                {
                    std::string extraNonceStr{ result.at(1).as_string().c_str() };
                    uint32_t const extraNonce2Size { common::boostJsonGetNumber<uint32_t>(result.at(2)) };
                    setExtraNonce(extraNonceStr, extraNonce2Size);
                    miningAuthorize();
                }
            }
            else
            {
                logErr() << "Subscribe failed : " << root;
            }
            break;
        }
        case stratum::Stratum::ID_MINING_AUTHORIZE:
        {
            if (true == root.at("error").is_null())
            {
                authenticated = root.at("result").as_bool();
                if (true == authenticated)
                {
                    logInfo() << "Successful login!";
                }
                else
                {
                    logErr() << "Fail to login : " << root;
                }
            }
            else
            {
                logErr() << "Authorize failed : " << root;
            }
            break;
        }
        default:
        {
            onShare(root, miningRequestID);
            break;
        }
    }
}


void stratum::StratumAutolykosV2::onMiningNotify(
    boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    auto const params{ root.at("params").as_array() };

    jobInfo.jobIDStr.assign(params.at(0).as_string().c_str());
    jobInfo.blockNumber = common::boostJsonGetNumber<uint64_t>(params.at(1));
    jobInfo.headerHash = algo::toHash<algo::hash256>(params.at(2).as_string().c_str());
    jobInfo.boundary =
        algo::toHash2<algo::hash256, algo::hash512>(
            algo::toLittleEndian<algo::hash512>(
                algo::decimalToHash<algo::hash512>(
                    params.at(6).as_string().c_str())));

    jobInfo.cleanJob = params.at(8).as_bool();

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.jobID = algo::toHash<algo::hash256>(jobInfo.jobIDStr);
    jobInfo.epoch = castU32(jobInfo.blockNumber);
    jobInfo.period = algo::autolykos_v2::computePeriod(jobInfo.epoch);

    ////////////////////////////////////////////////////////////////////////////
    logDebug() << jobInfo;

    ////////////////////////////////////////////////////////////////////////////
    updateJob();
}


void stratum::StratumAutolykosV2::onMiningSetDifficulty(
    boost::json::object const& root)
{
    auto const params{ root.at("params").as_array() };
    double const difficulty{ common::boostJsonGetNumber<double>(params.at(0)) };

    jobInfo.boundary = algo::toHash256(difficulty);
    jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
    if (jobInfo.boundaryU64 < jobInfo.targetBits)
    {
        jobInfo.boundaryU64 = jobInfo.targetBits;
    }

    logInfo() << "Target: " << std::hex << jobInfo.boundaryU64;
}


void stratum::StratumAutolykosV2::onMiningSetTarget(
    boost::json::object const& root)
{
    // https://github.com/mhssamadani/ErgoStratumServer
    logErr() << "mining.set_target unimplemented!";
}


void stratum::StratumAutolykosV2::onMiningSetExtraNonce(
    boost::json::object const& root)
{
    // https://github.com/mhssamadani/ErgoStratumServer
    logErr() << "mining.set_extranonce unimplemented!";
}


void stratum::StratumAutolykosV2::miningSubmit(
    uint32_t const deviceId,
    boost::json::array const& params)
{
    UNIQUE_LOCK(mtxSubmit);

    static thread_local uint32_t shareId { 1000u };
    boost::json::object root;
    root["id"] = shareId + (1000u * deviceId);
    root["method"] = "mining.submit";
    root["params"] = boost::json::array{ wallet + workerName};
    root["params"].as_array().push_back(params.at(0));
    root["params"].as_array().push_back(params.at(1));
    root["params"].as_array().push_back("");
    root["params"].as_array().push_back(params.at(2));


    send(root);
    ++shareId;
}
