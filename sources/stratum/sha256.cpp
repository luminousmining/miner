#include <string>
#include <variant>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <common/app.hpp>
#include <common/boost_utils.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/sha256.hpp>


void stratum::StratumSha256::onResponse(
    boost::json::object const& root)
{
    auto const miningRequestID{ common::boostJsonGetNumber<uint32_t>(root.at("id")) };

    switch(miningRequestID)
    {
        case stratum::Stratum::ID_MINING_SUBSCRIBE:
        {
            if (true == root.at("error").is_null())
            {
                boost::json::array const& result(root.at("result").as_array());
                if (false == result.empty())
                {
                    for (auto const& methods : result.at(0).as_array())
                    {
                        using namespace std::string_literals;
                        std::string const method { methods.at(0).as_string().c_str() };
                        if ("mining.notify"s == method)
                        {
                            sessionId.assign(methods.at(1).as_string().c_str());
                        }
                    }
                    setExtraNonce(result.at(1).as_string().c_str());

                    extraNonce2Size = common::boostJsonGetNumber<uint32_t>(result.at(2));

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


void stratum::StratumSha256::onMiningNotify(
    boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    boost::json::array const& params(root.at("params").as_array());

    std::string jobId { params.at(0).as_string().c_str() };
    jobInfo.headerHash = algo::toHash256(params.at(1).as_string().c_str());
    jobInfo.coinb1 = algo::toHash1024(params.at(2).as_string().c_str());
    jobInfo.coinb2 = algo::toHash<algo::hash2048>(params.at(3).as_string().c_str());
    boost::json::array merkletree(params.at(4).as_array());
    jobInfo.epoch = common::boostJsonGetNumber<int32_t>(params.at(5));
    jobInfo.targetBits = std::strtoul(params.at(6).as_string().c_str(), nullptr, 16);
    jobInfo.blockNumber = common::boostJsonGetNumber<uint64_t>(params.at(7));
    jobInfo.cleanJob = params.at(8).as_bool();

    jobInfo.jobIDStr.assign(jobId);
    jobInfo.jobID = algo::toHash256(jobId);
    for (size_t i { 0u }; i < merkletree.size(); ++i)
    {
        auto const& hash { merkletree.at(i) };
        jobInfo.merkletree[i] = algo::toHash256(hash.as_string().c_str());
    }

    ////////////////////////////////////////////////////////////////////////////
    updateJob();
}


void stratum::StratumSha256::onMiningSetDifficulty(
    boost::json::object const& root)
{
    boost::json::array const& params(root.at("params").as_array());

    // TODO : SHA256
    //double const difficulty{ common::boostJsonGetNumber<double>(params.at(0)) };

    logInfo() << "Target: " << std::hex << jobInfo.boundaryU64;
}


void stratum::StratumSha256::onMiningSetTarget(
    [[maybe_unused]] boost::json::object const& root)
{
    logErr() << "mining.set_target does not implement";
}


void stratum::StratumSha256::onMiningSetExtraNonce(
    [[maybe_unused]] boost::json::object const& root)
{
    logErr() << "mining.set_target does not implement";
}


void stratum::StratumSha256::miningSubscribe()
{
    auto const softwareName
    {
        "luminousminer/"
        + std::to_string(common::VERSION_MAJOR)
        + "."
        + std::to_string(common::VERSION_MINOR)
    };

    boost::json::object root;
    root["id"] = stratum::Stratum::ID_MINING_SUBSCRIBE;
    root["method"] = "mining.subscribe";
    root["params"] = boost::json::array{ softwareName };

    send(root);
}


void stratum::StratumSha256::miningSubmit(
    uint32_t const deviceId,
    boost::json::array const& params)
{
    UNIQUE_LOCK(mtxSubmit);

    boost::json::object root;
    root["id"] = (deviceId + 1u) * stratum::Stratum::OVERCOM_NONCE;
    root["method"] = "mining.submit";
    root["params"] = boost::json::array
    {
        wallet + "." + workerName,              // login.workername
        params.at(0),                           // JobID
        params.at(1),                           // nonce
        "0x" + algo::toHex(jobInfo.headerHash), // header
        params.at(2)                            // mix
    };

    send(root);
}
