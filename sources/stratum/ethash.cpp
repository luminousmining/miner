#include <string>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/ethash/ethash.hpp>
#include <common/boost_utils.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/ethash.hpp>


void stratum::StratumEthash::onResponse(
    boost::json::object const& root)
{
    auto const miningRequestID{ common::boostJsonGetNumber<uint32_t>(root.at("id")) };

    switch(miningRequestID)
    {
        case stratum::Stratum::ID_MINING_SUBSCRIBE:
        {
            if (   false == root.contains("error")
                || true == root.at("error").is_null())
            {
                auto result{ root.at("result").as_array() };
                if (false == result.empty())
                {
                    std::string extraNonceStr{ result.at(1).as_string().c_str() };
                    setExtraNonce(extraNonceStr);
                    jobInfo.targetBits = std::strtoul(extraNonceStr.c_str(), nullptr, 16);;
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
            if (   false == root.contains("error")
                || true == root.at("error").is_null())
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


void stratum::StratumEthash::onMiningNotify(
    boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    boost::json::array const& params(root.at("params").as_array());

    ////////////////////////////////////////////////////////////////////////////
    std::string const jobID{ params.at(0).as_string().c_str() };
    jobInfo.seedHash = algo::toHash256(params.at(1).as_string().c_str());
    jobInfo.headerHash = algo::toHash256(params.at(2).as_string().c_str());
    jobInfo.cleanJob = params.at(3).as_bool();

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.jobIDStr.assign(jobID);
    jobInfo.jobID = algo::toHash256(jobID);

    ////////////////////////////////////////////////////////////////////////////
    int32_t const epoch{ algo::ethash::ContextGenerator::instance().findEpoch(jobInfo.seedHash, maxEpochNumber) };
    if (-1 != epoch)
    {
        jobInfo.epoch = epoch;
    }

    ////////////////////////////////////////////////////////////////////////////
    updateJob();
}


void stratum::StratumEthash::onMiningSetDifficulty(
    boost::json::object const& root)
{
    boost::json::array const& params(root.at("params").as_array());
    double const currentDifficulty { common::boostJsonGetNumber<double>(params.at(0)) };

    double difficulty
    {
        currentDifficulty <= 2.0
            ? currentDifficulty
            : ((currentDifficulty * 2.0) / 8589934592.0)
    };

    difficulty = common::min_limit(difficulty, 0.0001);
    difficulty = common::max_limit(difficulty, 2.0);

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


void stratum::StratumEthash::onMiningSetTarget(
    boost::json::object const& root)
{
    boost::json::array const& params(root.at("params").as_array());
    std::string const boundary{ params.at(0).as_string().c_str() };

    jobInfo.boundary = algo::toHash256(boundary);
    jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
    if (jobInfo.boundaryU64 < jobInfo.targetBits)
    {
        jobInfo.boundaryU64 = jobInfo.targetBits;
    }

    logInfo() << "Target: " << std::hex << jobInfo.boundaryU64;
}


void stratum::StratumEthash::onMiningSetExtraNonce(
    boost::json::object const& root)
{
    boost::json::array const& params(root.at("params").as_array());
    std::string const extraNonceStr{ params.at(0).as_string().c_str() };
    setExtraNonce(extraNonceStr);
}


void stratum::StratumEthash::miningSubmit(
    uint32_t const deviceId,
    boost::json::array const& params)
{
    using namespace std::string_literals;

    UNIQUE_LOCK(mtxSubmit);

    boost::json::object root;
    root["id"] = (deviceId + 1u) * stratum::Stratum::OVERCOM_NONCE;
    root["method"] = "mining.submit";
    root["params"] = boost::json::array
    {
        wallet + "."s + workerName, // login.workername
        params.at(0),               // JobID
        params.at(1),               // nonce
    };

    send(root);
}
