#include <string>
#include <random>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/ethash/ethash.hpp>
#include <common/boost_utils.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/progpow.hpp>


void stratum::StratumProgPOW::onResponse(
    boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    switch(stratumType)
    {
        case stratum::STRATUM_TYPE::ETHEREUM_V1:
        {
            onResponseEthereumV1(root);
            break;
        }
        case stratum::STRATUM_TYPE::ETHEREUM_V2:
        {
            onResponseEthereumV2(root);
            break;
        }
        case stratum::STRATUM_TYPE::ETHPROXY:
        {
            onResponseEthProxy(root);
            break;
        }
    }
}


void stratum::StratumProgPOW::onResponseEthereumV1(boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    auto const miningRequestID{ common::boostJsonGetNumber<uint32_t>(root.at("id")) };

    ////////////////////////////////////////////////////////////////////////////
    switch(miningRequestID)
    {
        case stratum::Stratum::ID_MINING_SUBSCRIBE:
        {
            ////////////////////////////////////////////////////////////////////
            if (   false == root.contains("error")
                || true == root.at("error").is_null())
            {
                boost::json::array const& result(root.at("result").as_array());
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

            ////////////////////////////////////////////////////////////////////
            break;
        }
        case stratum::Stratum::ID_MINING_AUTHORIZE:
        {
            ////////////////////////////////////////////////////////////////////
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

            ////////////////////////////////////////////////////////////////////
            break;
        }
        default:
        {
            ////////////////////////////////////////////////////////////////////
            onShare(root, miningRequestID);

            ////////////////////////////////////////////////////////////////////
            break;
        }
    }
}


void stratum::StratumProgPOW::onResponseEthereumV2(boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    auto const miningRequestID{ common::boostJsonGetNumber<uint32_t>(root.at("id")) };
    auto const requestID{ static_cast<stratum::ETHEREUM_V2_ID>(miningRequestID) };

    ////////////////////////////////////////////////////////////////////////////
    switch(requestID)
    {
        case stratum::ETHEREUM_V2_ID::MINING_HELLO:
        {
            ////////////////////////////////////////////////////////////////////
            if (   false == root.contains("error")
                || true == root.at("error").is_null())
            {
                auto const& result{ root.at("result").as_object() };

                std::string const encoding{ common::boostGetString(result, "encoding"s) };
                std::string const node{ common::boostGetString(result, "node"s) };
                std::string const proto{ common::boostGetString(result, "proto"s) };
                maxErrors = common::boostJsonGetNumber<uint32_t>(result, "maxerrors"s);
                resume = common::boostJsonGetNumber<uint32_t>(result, "resume"s);
                timeout = common::boostJsonGetNumber<uint32_t>(result, "timeout"s);

                if (   "plain"s == encoding
                    && false == node.empty()
                    && false == proto.empty()
                    && 0u < maxErrors)
                {
                    miningAuthorize();
                }
            }
            else
            {
                logErr() << "Hello failed: " << root;
            }

            ////////////////////////////////////////////////////////////////////
            break;
        }
        case stratum::ETHEREUM_V2_ID::MINING_AUTHORIZE:
        {
            ////////////////////////////////////////////////////////////////////
            if (   false == root.contains("error")
                || true == root.at("error").is_null())
            {
                workerID = common::boostGetString(root, "result");
                authenticated = true;
                doLoopTimeout();
            }
            else
            {
                authenticated = false;
                logErr() << "Authorize failed: " << root;
            }

            ////////////////////////////////////////////////////////////////////
            break;
        }
        default:
        {
            ////////////////////////////////////////////////////////////////////
            onShare(root, miningRequestID);

            ////////////////////////////////////////////////////////////////////
            break;
        }
    }
}


void stratum::StratumProgPOW::onResponseEthProxy(boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    uint32_t miningRequestID{ 0u };
    stratum::ETHPROXY_ID requestID{ stratum::ETHPROXY_ID::EMPTY };
    if (true == common::boostJsonContains(root, "id"))
    {
        miningRequestID = common::boostJsonGetNumber<uint32_t>(root, "id");
        requestID = static_cast<stratum::ETHPROXY_ID>(miningRequestID);
    }

    switch(requestID)
    {
        case stratum::ETHPROXY_ID::SUBMITLOGIN:
        {
            ////////////////////////////////////////////////////////////////////
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
                ethGetWork();
            }

            ////////////////////////////////////////////////////////////////////
            break;
        }
        case stratum::ETHPROXY_ID::EMPTY: /*stratum::ETHPROXY_ID::GETWORK */
        case stratum::ETHPROXY_ID::GETWORK:
        {
            ////////////////////////////////////////////////////////////////////
            onMiningNotify(root);

            ////////////////////////////////////////////////////////////////////
            break;
        }
        default:
        {
            ////////////////////////////////////////////////////////////////////
            onShare(root, miningRequestID);

            ////////////////////////////////////////////////////////////////////
            break;
        }
    }
}


void stratum::StratumProgPOW::miningSubmit(
    uint32_t const deviceId,
    boost::json::array const& params)
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;
    UNIQUE_LOCK(mtxSubmit);

    ////////////////////////////////////////////////////////////////////////////
    boost::json::object root;
    switch(stratumType)
    {
        case stratum::STRATUM_TYPE::ETHEREUM_V1:
        {
            ////////////////////////////////////////////////////////////////////
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

            ////////////////////////////////////////////////////////////////////
            break;
        }
        case stratum::STRATUM_TYPE::ETHEREUM_V2:
        {
            ////////////////////////////////////////////////////////////////////
            root["id"] = (deviceId + 1u) * stratum::Stratum::OVERCOM_NONCE;
            root["method"] = "mining.submit";
            root["params"] = boost::json::array
            {
                params.at(0), // JobID
                params.at(1), // nonce
                params.at(2)  // workerID
            };

            ////////////////////////////////////////////////////////////////////
            break;
        }
        case stratum::STRATUM_TYPE::ETHPROXY:
        {
            ////////////////////////////////////////////////////////////////////
            root["id"] = (deviceId + 1u) * stratum::Stratum::OVERCOM_NONCE;
            root["method"] = "eth_submitWork";
            root["params"] = boost::json::array
            {
                params.at(0),                           // nonce
                "0x" + algo::toHex(jobInfo.headerHash), // header
                params.at(1)                            // hash
            };

            ////////////////////////////////////////////////////////////////////
            break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    send(root);
}


void stratum::StratumProgPOW::onMiningNotify(
    boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    switch(stratumType)
    {
        case stratum::STRATUM_TYPE::ETHEREUM_V1:
        {
            ////////////////////////////////////////////////////////////////////
            boost::json::array const& params(root.at("params").as_array());

            ////////////////////////////////////////////////////////////////////
            std::string const jobID{ params.at(0).as_string().c_str() };
            jobInfo.headerHash = algo::toHash256(params.at(1).as_string().c_str());
            jobInfo.seedHash = algo::toHash256(params.at(2).as_string().c_str());
            jobInfo.boundary = algo::toHash256(params.at(3).as_string().c_str());
            jobInfo.cleanJob = params.at(4).as_bool();
            jobInfo.blockNumber = common::boostJsonGetNumber<uint64_t>(params.at(5));
            jobInfo.targetBits = std::strtoul(params.at(6).as_string().c_str(), nullptr, 16);

            ////////////////////////////////////////////////////////////////////
            jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
            jobInfo.jobID = algo::toHash256(jobID);
            jobInfo.jobIDStr.assign(jobID);
            jobInfo.period = jobInfo.blockNumber / maxPeriod;

            ////////////////////////////////////////////////////////////////////
            if (jobInfo.boundaryU64 < jobInfo.targetBits)
            {
                jobInfo.boundaryU64 = jobInfo.targetBits;
            }

            ////////////////////////////////////////////////////////////////////
            int32_t epoch{ 0 };
            if (jobInfo.blockNumber > 0ull)
            {
                epoch = cast32(jobInfo.blockNumber / castU64(maxEpochLength));
            }
            else
            {
                epoch = algo::ethash::ContextGenerator::instance().findEpoch(jobInfo.seedHash, maxEthashEpoch);
            }
            if (-1 != epoch)
            {
                jobInfo.epoch = epoch;
            }

            ////////////////////////////////////////////////////////////////////
            break;
        }
        case stratum::STRATUM_TYPE::ETHEREUM_V2:
        {
            ////////////////////////////////////////////////////////////////////
            boost::json::array const& params(root.at("params").as_array());

            ////////////////////////////////////////////////////////////////////
            std::string const jobID{ params.at(0).as_string().c_str() };
            std::string const blockHeight{ params.at(1).as_string().c_str() };
            std::string const header{ params.at(2).as_string().c_str() };

            ////////////////////////////////////////////////////////////////////
            jobInfo.headerHash = algo::toHash256(header);
            jobInfo.jobID = algo::toHash256(jobID);
            jobInfo.jobIDStr.assign(jobID);
            jobInfo.blockNumber = std::strtoul(blockHeight.c_str(), nullptr, 16);
            jobInfo.period = jobInfo.blockNumber / maxPeriod;

            ////////////////////////////////////////////////////////////////////
            break;
        }
        case stratum::STRATUM_TYPE::ETHPROXY:
        {
            ////////////////////////////////////////////////////////////////////
            auto const& params{ root.at("result").as_array() };

            ////////////////////////////////////////////////////////////////////
            std::string const jobID{ common::boostGetString(params, 0) };
            jobInfo.headerHash = algo::toHash256(common::boostGetString(params, 0));
            jobInfo.seedHash = algo::toHash256(common::boostGetString(params, 1));
            jobInfo.boundary = algo::toHash256(common::boostGetString(params, 2));
            std::string const blockNumber{ common::boostGetString(params, 3) };
            logInfo() << "blockNumber: " << blockNumber;

            ////////////////////////////////////////////////////////////////////
            jobInfo.jobID = algo::toHash256(jobID);
            jobInfo.jobIDStr.assign(jobID);
            jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
            jobInfo.blockNumber = std::strtoull(blockNumber.c_str(), nullptr, 16);
            jobInfo.period = jobInfo.blockNumber / maxPeriod;

            ////////////////////////////////////////////////////////////////////
            std::random_device random;
            std::mt19937 gen{ random() };
            std::uniform_int_distribution<uint64_t> dis(100000, 999999);
            uint64_t const nonce{ dis(gen) };
            std::stringstream ss;
            ss << std::hex << std::uppercase << std::setw(6) << std::setfill('0') << nonce;
            setExtraNonce(ss.str());

            ////////////////////////////////////////////////////////////////////
            int32_t epoch{ 0 };
            if (jobInfo.blockNumber > 0ull)
            {
                epoch = cast32(jobInfo.blockNumber / castU64(maxEpochLength));
            }
            else
            {
                epoch = algo::ethash::ContextGenerator::instance().findEpoch(jobInfo.seedHash, maxEthashEpoch);
            }
            if (-1 != epoch)
            {
                jobInfo.epoch = epoch;
            }

            ////////////////////////////////////////////////////////////////////
            break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    updateJob();
}


void stratum::StratumProgPOW::onMiningSetDifficulty(
    boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    boost::json::array const& params(root.at("params").as_array());
    double const difficulty{ common::boostJsonGetNumber<double>(params.at(0)) };

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.boundary = algo::toHash256(difficulty);
    jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
    if (jobInfo.boundaryU64 < jobInfo.targetBits)
    {
        jobInfo.boundaryU64 = jobInfo.targetBits;
    }

    ////////////////////////////////////////////////////////////////////////////
    logInfo()
        << "Target: "
        << std::hex << jobInfo.boundaryU64
        << std::dec << " (" << difficulty << ")";
}


void stratum::StratumProgPOW::onMiningSetTarget(
    boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    boost::json::array const& params(root.at("params").as_array());
    auto boundary{ params.at(0).as_string().c_str() };

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.boundary = algo::toHash256(boundary);
    jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
    if (jobInfo.boundaryU64 < jobInfo.targetBits)
    {
        jobInfo.boundaryU64 = jobInfo.targetBits;
    }

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Target: " << std::hex << jobInfo.boundaryU64;
}


void stratum::StratumProgPOW::onMiningSetExtraNonce(
    boost::json::object const& root)
{
    boost::json::array const& params(root.at("params").as_array());
    std::string const extraNonceStr{ params.at(0).as_string().c_str() };
    setExtraNonce(extraNonceStr);
}


void stratum::StratumProgPOW::onMiningSet(
    boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    if (stratum::STRATUM_TYPE::ETHEREUM_V2 != stratumType)
    {
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    boost::json::object const& params(root.at("params").as_object());

    ////////////////////////////////////////////////////////////////////////////
    std::string const algo{ common::boostGetString(params, "algo") };
    std::string const epoch{ common::boostGetString(params, "epoch") };
    std::string const extraNonce{ common::boostGetString(params, "extranonce") };
    std::string const target{ common::boostGetString(params, "target") };

    ////////////////////////////////////////////////////////////////////////////
    setExtraNonce(extraNonce);

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.epoch = std::stoull(epoch);
    jobInfo.boundary = algo::toHash256(target);
    jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
    if (jobInfo.boundaryU64 < jobInfo.targetBits)
    {
        jobInfo.boundaryU64 = jobInfo.targetBits;
    }
}
