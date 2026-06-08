#include <string>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/kheavyhash/stratum_math.hpp>
#include <common/boost_utils.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/kheavyhash.hpp>


void stratum::StratumKHeavyHash::onResponse(boost::json::object const& root)
{
    auto const miningRequestID{ common::boostJsonGetNumber<uint32_t>(root.at("id")) };

    switch (miningRequestID)
    {
        case stratum::Stratum::ID_MINING_SUBSCRIBE:
        {
            // Subscribe ack. Some bridges return the extranonce here as an array
            // element; if present and a string, adopt it. Otherwise it arrives via
            // mining.set_extranonce.
            if (true == root.contains("result") && true == root.at("result").is_array())
            {
                boost::json::array const& result(root.at("result").as_array());
                if (false == result.empty() && true == result.at(0).is_string())
                {
                    setExtraNonce(result.at(0).as_string().c_str());
                }
            }
            break;
        }
        case stratum::Stratum::ID_MINING_AUTHORIZE:
        {
            if (false == root.contains("error") || true == root.at("error").is_null())
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


void stratum::StratumKHeavyHash::onUnknownMethod(boost::json::object const& root)
{
    std::string const method{ common::boostGetString(root, "method") };

    if ("mining.authorize" == method)
    {
        onResponse(root);
    }
}


void stratum::StratumKHeavyHash::onMiningNotify(boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    // params = [ jobIdStr, [u64_0, u64_1, u64_2, u64_3], timestamp ]
    boost::json::array const& params(root.at("params").as_array());
    if (3u > params.size())
    {
        logErr() << "mining.notify: malformed params " << root;
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.jobIDStr.assign(params.at(0).as_string().c_str());

    boost::json::array const& words(params.at(1).as_array());
    if (4u != words.size())
    {
        logErr() << "mining.notify: expected 4 pre_pow words " << root;
        return;
    }
    uint64_t prePowWords[4]{};
    for (uint32_t i{ 0u }; i < 4u; ++i)
    {
        prePowWords[i] = common::boostJsonGetNumber<uint64_t>(words.at(i));
    }
    kheavyhash::Hash256 const prePow{ kheavyhash::prePowFromWords(prePowWords) };
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        jobInfo.headerHash.ubytes[i] = prePow[i];
    }

    jobInfo.timestamp = common::boostJsonGetNumber<uint64_t>(params.at(2));

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.jobID = algo::toHash256(jobInfo.jobIDStr);

    ////////////////////////////////////////////////////////////////////////////
    // Restart the per-job nonce sweep from the (extranonce) base.
    jobInfo.nonce = jobInfo.startNonce;

    ////////////////////////////////////////////////////////////////////////////
    // kHeavyHash is not memory-hard: there is no DAG/epoch. Pin epoch to a single
    // constant so the resolver's updateMemory (buffer alloc + kernel build) runs
    // once; subsequent jobs differ only by headerHash and trigger updateConstants
    // (matrix/header/target re-upload) -- see DeviceManager::onUpdateJob.
    jobInfo.epoch = 1;

    ////////////////////////////////////////////////////////////////////////////
    updateJob();
}


void stratum::StratumKHeavyHash::onMiningSetDifficulty(boost::json::object const& root)
{
    boost::json::array const& params(root.at("params").as_array());
    double const              difficulty{ common::boostJsonGetNumber<double>(params.at(0)) };

    kheavyhash::Hash256 const target{ kheavyhash::difficultyToTargetLe(difficulty) };
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        jobInfo.boundary.ubytes[i] = target[i];
    }

    logInfo() << "Difficulty: " << difficulty;
}


void stratum::StratumKHeavyHash::onMiningSetExtraNonce(boost::json::object const& root)
{
    boost::json::array const& params(root.at("params").as_array());
    if (false == params.empty() && true == params.at(0).is_string())
    {
        setExtraNonce(params.at(0).as_string().c_str());
        logInfo() << "Nonce start: " << std::hex << jobInfo.startNonce << std::dec;
    }
}


void stratum::StratumKHeavyHash::miningSubscribe()
{
    // Send mining.subscribe (so the bridge records a non-BzMiner agent and keeps
    // the NORMAL job encoding), then authorize.
    stratum::Stratum::miningSubscribe();
    miningAuthorize();
}


void stratum::StratumKHeavyHash::miningSubmit(uint32_t const deviceId, boost::json::array const& params)
{
    using namespace std::string_literals;

    UNIQUE_LOCK(mtxSubmit);

    // params from the resolver = [ jobIdStr, nonceHex ].
    boost::json::object root;
    root["id"] = (deviceId + 1u) * stratum::Stratum::OVERCOM_NONCE;
    root["method"] = "mining.submit";
    root["params"] = boost::json::array{ wallet + "."s + workerName, params.at(0), params.at(1) };

    send(root);
}
