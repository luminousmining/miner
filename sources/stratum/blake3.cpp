#include <limits>
#include <string>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <common/boost_utils.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/blake3.hpp>


stratum::StratumBlake3::StratumBlake3()
{
    stratumName.assign("AlephiumStratum/1.0.0");
}


void stratum::StratumBlake3::onResponse(
    [[maybe_unused]] boost::json::object const& root)
{
    auto const miningRequestID{ common::boostJsonGetNumber<uint32_t>(root.at("id")) };

    switch(miningRequestID)
    {
        case stratum::Stratum::ID_MINING_AUTHORIZE:
        {
            if (   false == root.contains("error")
                || true == root.at("params").as_bool())
            {
                authenticated = true;
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


void stratum::StratumBlake3::onUnknowMethod(
    boost::json::object const& root)
{
    std::string const method{ common::boostGetString(root, "method") };

    if ("mining.authorize" == method)
    {
        onResponse(root);
    }
}


void stratum::StratumBlake3::onMiningNotify(
    boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    boost::json::array const& params(root.at("params").as_array());
    size_t const length { params.size() };
    boost::json::object jobParam { params[length - 1].as_object() };

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.jobIDStr.assign(jobParam.at("jobId").as_string().c_str());
    jobInfo.fromGroup = common::boostJsonGetNumber<uint32_t>(jobParam.at("fromGroup"));
    jobInfo.toGroup = common::boostJsonGetNumber<uint32_t>(jobParam.at("toGroup"));
    jobInfo.targetBlob = algo::toHash256(common::boostGetString(jobParam, "targetBlob"));
    jobInfo.headerBlob = algo::toHash<algo::hash3072>(
        common::boostGetString(jobParam, "headerBlob"),
        algo::HASH_SHIFT::RIGHT);

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.jobID = algo::toHash256(jobInfo.jobIDStr);

    ////////////////////////////////////////////////////////////////////////////
    // fake data
    if (-1 == jobInfo.epoch)
    {
        jobInfo.epoch = 1u;
    }
    else
    {
        ++jobInfo.epoch;
        if (jobInfo.epoch >= std::numeric_limits<int32_t>::max())
        {
            jobInfo.epoch = 1u;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    updateJob();
}


void stratum::StratumBlake3::onMiningSetDifficulty(
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

    logInfo()
        << "Target: "
        << std::hex << jobInfo.boundaryU64
        << std::dec << " (" << difficulty << ")";
}


void stratum::StratumBlake3::onMiningSetExtraNonce(
    boost::json::object const& root)
{
    boost::json::array const& params(root.at("params").as_array());
    std::string const extraNonceStr{ params.at(0).as_string().c_str() };
    setExtraNonce(extraNonceStr);

    logInfo() << "Nonce start: " << std::hex << jobInfo.startNonce;
}


void stratum::StratumBlake3::miningSubscribe()
{
    // ignore mining.subscribe
    // can use directly mining.authorize
    miningAuthorize();
}


void stratum::StratumBlake3::miningSubmit(
    [[maybe_unused]] uint32_t const deviceId,
    [[maybe_unused]] boost::json::object const& params)
{
    using namespace std::string_literals;

    UNIQUE_LOCK(mtxSubmit);

    boost::json::object root;
    root["id"] = (deviceId + 1u) * stratum::Stratum::OVERCOM_NONCE;
    root["method"] = "mining.submit";
    root["params"] = boost::json::object
    {
        { "jobId",     common::boostGetString(params, "jobId") },
        { "fromGroup", common::boostJsonGetNumber<uint32_t>(params.at("fromGroup")) },
        { "toGroup",   common::boostJsonGetNumber<uint32_t>(params.at("toGroup")) },
        { "nonce",     common::boostGetString(params, "nonce") },
        { "worker",    wallet + "."s + workerName }
    };

    //logDebug() << root;

    send(root);
}
