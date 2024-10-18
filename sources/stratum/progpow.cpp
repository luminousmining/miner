#include <string>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/ethash/ethash.hpp>
#include <common/boost_utils.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/progpow.hpp>


stratum::StratumProgPOW::StratumProgPOW()
{
    stratumName.assign("EthereumStratum/1.0.0");
}


void stratum::StratumProgPOW::onResponse(
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
                    setExtraNonce(extraNonceStr);
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


void stratum::StratumProgPOW::miningSubmit(
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
        wallet + "."s + workerName,             // login.workername
        params.at(0),                           // JobID
        params.at(1),                           // nonce
        "0x" + algo::toHex(jobInfo.headerHash), // header
        params.at(2)                            // mix
    };

    send(root);
}


void stratum::StratumProgPOW::onMiningNotify(
    boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    auto const params{ root.at("params").as_array() };

    ////////////////////////////////////////////////////////////////////////////
    std::string const jobID{ params.at(0).as_string().c_str() };
    jobInfo.headerHash = algo::toHash256(params.at(1).as_string().c_str());
    jobInfo.seedHash = algo::toHash256(params.at(2).as_string().c_str());
    jobInfo.boundary = algo::toHash256(params.at(3).as_string().c_str());
    jobInfo.cleanJob = params.at(4).as_bool();
    jobInfo.blockNumber = common::boostJsonGetNumber<uint64_t>(params.at(5));
    jobInfo.targetBits = std::strtoul(params.at(6).as_string().c_str(), nullptr, 16);

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
    jobInfo.jobID = algo::toHash256(jobID);
    jobInfo.jobIDStr.assign(jobID);
    jobInfo.period = jobInfo.blockNumber / maxPeriod;

    ////////////////////////////////////////////////////////////////////////////
    if (jobInfo.boundaryU64 < jobInfo.targetBits)
    {
        jobInfo.boundaryU64 = jobInfo.targetBits;
    }

    ////////////////////////////////////////////////////////////////////////////
    int32_t epoch;
    if (jobInfo.blockNumber > 0ull)
    {
        epoch = cast32(jobInfo.blockNumber / castU64(maxEpochLength));
    }
    else
    {
        epoch = algo::ethash::findEpoch(jobInfo.seedHash, maxEthashEpoch);
    }
    if (-1 != epoch)
    {
        jobInfo.epoch = epoch;
    }

    ////////////////////////////////////////////////////////////////////////////
    updateJob();
}


void stratum::StratumProgPOW::onMiningSetDifficulty(
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


void stratum::StratumProgPOW::onMiningSetTarget(
    boost::json::object const& root)
{
    auto const params{ root.at("params").as_array() };
    auto boundary{ params.at(0).as_string().c_str() };
    
    jobInfo.boundary = algo::toHash256(boundary);
    jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
    if (jobInfo.boundaryU64 < jobInfo.targetBits)
    {
        jobInfo.boundaryU64 = jobInfo.targetBits;
    }

    logInfo() << "Target: " << std::hex << jobInfo.boundaryU64;
}


void stratum::StratumProgPOW::onMiningSetExtraNonce(
    boost::json::object const& root)
{
    auto const params{ root.at("params").as_array() };
    std::string const extraNonceStr{ params.at(0).as_string().c_str() };
    setExtraNonce(extraNonceStr);
}
