#include <string>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <common/app.hpp>
#include <common/boost_utils.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/random_x.hpp>


void stratum::StratumRandomX::onConnect()
{
    logInfo() << "Stratum connected!";

    if (true == wallet.empty())
    {
        logErr() << "Cannot connect: wallet is empty";
        return;
    }

    auto const softwareName{ "luminousminer/" + std::to_string(common::VERSION_MAJOR) + "."
                             + std::to_string(common::VERSION_MINOR) };

    boost::json::object root;
    root["id"]     = stratum::Stratum::ID_MINING_SUBSCRIBE;
    root["method"] = "login";
    root["params"] = boost::json::object{
        { "login", wallet },
        { "pass", password.empty() ? std::string("x") : password },
        { "agent", softwareName },
        { "algo", boost::json::array{ "rx/0" } }
    };

    send(root);
}


void stratum::StratumRandomX::onResponse(boost::json::object const& root)
{
    auto const miningRequestID{ common::boostJsonGetNumber<uint32_t>(root.at("id")) };

    switch (miningRequestID)
    {
        case stratum::Stratum::ID_MINING_SUBSCRIBE:
        {
            if (false == root.contains("error") || true == root.at("error").is_null())
            {
                boost::json::object const& result{ root.at("result").as_object() };
                minerID.assign(result.at("id").as_string().c_str());
                authenticated = true;
                logInfo() << "Successful login! miner_id=" << minerID;
                parseJob(result.at("job").as_object());
            }
            else
            {
                logErr() << "Login failed: " << root;
            }
            break;
        }
        default:
        {
            using namespace std::string_literals;

            bool isValid{ true };

            if (false == common::boostJsonContains(root, "result") || true == root.at("result").is_null())
            {
                isValid = false;
            }
            else if (true == root.at("result").is_object())
            {
                auto const& resultObj{ root.at("result").as_object() };
                if (false == resultObj.contains("status")
                    || "OK"s != resultObj.at("status").as_string().c_str())
                {
                    isValid = false;
                }
            }
            else
            {
                isValid = false;
            }

            if (false == isValid)
            {
                logErr() << root;
            }

            if (nullptr != shareStatus)
            {
                shareStatus(isValid, miningRequestID, uuid);
            }
            break;
        }
    }
}


void stratum::StratumRandomX::onMiningNotify([[maybe_unused]] boost::json::object const& root)
{
}


void stratum::StratumRandomX::onMiningSetDifficulty([[maybe_unused]] boost::json::object const& root)
{
}


void stratum::StratumRandomX::onUnknownMethod(boost::json::object const& root)
{
    std::string const method{ root.at("method").as_string().c_str() };

    if ("job" == method)
    {
        parseJob(root.at("params").as_object());
    }
}


void stratum::StratumRandomX::miningSubmit(uint32_t const deviceId, boost::json::array const& params)
{
    UNIQUE_LOCK(mtxSubmit);

    boost::json::object root;
    root["id"]     = (deviceId + 1u) * stratum::Stratum::OVERCOM_NONCE;
    root["method"] = "submit";
    root["params"] = boost::json::object{
        { "id", minerID },
        { "job_id", params.at(0) },
        { "nonce", params.at(1) },
        { "result", params.at(2) }
    };

    send(root);
}


void stratum::StratumRandomX::parseJob(boost::json::object const& job)
{
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    // job_id: copy raw string bytes into jobID (avoids hex-parse issues)
    jobInfo.jobIDStr.assign(common::boostGetString(job, "job_id"));
    uint32_t const jobIdLen{ static_cast<uint32_t>(jobInfo.jobIDStr.size()) };
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        jobInfo.jobID.ubytes[i] = (i < jobIdLen) ? static_cast<uint8_t>(jobInfo.jobIDStr[i]) : 0u;
    }

    ////////////////////////////////////////////////////////////////////////////
    // blob: 77 bytes, fill ubytes[0..76] byte-by-byte (toHash fills from END — unusable here)
    std::string const blobHex{ common::boostGetString(job, "blob") };
    for (uint32_t i{ 0u }; i < 77u; ++i)
    {
        std::string const byteStr{ blobHex.substr(i * 2u, 2u) };
        jobInfo.headerBlob.ubytes[i] = static_cast<uint8_t>(std::stoul(byteStr, nullptr, 16));
    }

    ////////////////////////////////////////////////////////////////////////////
    // seed_hash: 32-byte hex → hash256
    jobInfo.seedHash = algo::toHash256(common::boostGetString(job, "seed_hash"));

    ////////////////////////////////////////////////////////////////////////////
    // target: 4-byte little-endian hex → uint32
    std::string const targetHex{ common::boostGetString(job, "target") };
    uint32_t targetBits{ 0u };
    for (uint32_t i{ 0u }; i < 4u; ++i)
    {
        std::string const byteStr{ targetHex.substr(i * 2u, 2u) };
        uint8_t const     byte{ static_cast<uint8_t>(std::stoul(byteStr, nullptr, 16)) };
        targetBits |= (static_cast<uint32_t>(byte) << (i * 8u));
    }
    jobInfo.targetBits = targetBits;

    ////////////////////////////////////////////////////////////////////////////
    // height
    jobInfo.blockNumber = common::boostJsonGetNumber<uint64_t>(job.at("height"));

    ////////////////////////////////////////////////////////////////////////////
    // boundary: targetBits stored big-endian in last 4 bytes, for isValidJob()
    jobInfo.boundary            = algo::hash256{};
    jobInfo.boundary.ubytes[28] = static_cast<uint8_t>((targetBits >> 24) & 0xFFu);
    jobInfo.boundary.ubytes[29] = static_cast<uint8_t>((targetBits >> 16) & 0xFFu);
    jobInfo.boundary.ubytes[30] = static_cast<uint8_t>((targetBits >>  8) & 0xFFu);
    jobInfo.boundary.ubytes[31] = static_cast<uint8_t>( targetBits        & 0xFFu);
    jobInfo.boundaryU64         = static_cast<uint64_t>(targetBits);

    ////////////////////////////////////////////////////////////////////////////
    // start nonce for this job
    jobInfo.nonce      = 1ull;
    jobInfo.startNonce = 1ull;

    ////////////////////////////////////////////////////////////////////////////
    updateJob();
}
