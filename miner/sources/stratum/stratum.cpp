#include <common/app.hpp>
#include <common/boost_utils.hpp>
#include <common/cast.hpp>
#include <common/config.hpp>
#include <algo/hash_utils.hpp>
#include <common/log/log.hpp>
#include <stratum/stratum.hpp>


stratum::Stratum::Stratum()
{
}


stratum::Stratum::~Stratum()
{
}


void stratum::Stratum::setCallbackUpdateJob(
    stratum::Stratum::callbackUpdateJob cbUpdateJob)
{
    if (nullptr != cbUpdateJob)
    {
        dispatchJob = cbUpdateJob;
    }
}


void stratum::Stratum::onReceive(
    std::string const& message)
{
    try
    {
        auto root{ boost::json::parse(message).as_object() };
        logDebug() << root;

        if (true == root.contains("method"))
        {
            onMethod(root);
        }
        else
        {
            onResponse(root);
        }
    }
    catch (boost::exception const& e)
    {
        logErr() << diagnostic_information(e);
    }
    catch (std::exception const& e)
    {
        logErr() << e.what();
    }
}


void stratum::Stratum::onMethod(
    boost::json::object const& root)
{
    std::string const method{ root.at("method").as_string().c_str() };

    if ("mining.notify" == method)
    {
        onMiningNotify(root);
    }
    else if ("mining.set_difficulty" == method)
    {
        onMiningSetDifficulty(root);
    }
    else if ("mining.set_target" == method)
    {
        onMiningSetTarget(root);
    }
    else if ("mining.set_extranonce" == method)
    {
        onMiningSetExtraNonce(root);
    }
    else if ("client.show_message" == method)
    {
        auto params{ root.at("params").as_array() };
        std::stringstream ss;
        for (auto msg : params)
        {
            ss << msg;
        }
        logWarn() << "Pool: " << ss.str();
    }
    else
    {
        logErr() << "Unknow[" << method << "]";
    }
}


void stratum::Stratum::onConnect()
{
    logInfo() << "Stratum connected!";
    common::Config const& config{ common::Config::instance() };

    if (   true == config.mining.wallet.empty()
        || true == config.mining.password.empty())
    {
        logErr()
            << "Cannot connect wallet[" << config.mining.wallet << "]"
            << " password[" << config.mining.password << "]";
        return;
    }

    miningSubscribe();
}


void stratum::Stratum::miningSubscribe()
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
    root["params"] = boost::json::array{ softwareName, stratum::STRATUM_VERSION };

    send(root);
}



void stratum::Stratum::miningAuthorize()
{
    boost::json::object root;
    root["id"] = stratum::Stratum::ID_MINING_AUTHORIZE;
    root["method"] = "mining.authorize";
    root["params"] = boost::json::array{ wallet + "." + workerName, password };

    send(root);
}


void stratum::Stratum::setExtraNonce(
    std::string const& paramExtraNonce)
{
    jobInfo.extraNonceSize = castU32(paramExtraNonce.size());
    jobInfo.extraNonce = std::strtoull(paramExtraNonce.c_str(), nullptr, 16);

    // Define the 0 counter to define nonce gap.
    size_t fill{ 16ull - paramExtraNonce.size() };
    std::string extraNonceFill{};
    extraNonceFill.assign(paramExtraNonce);
    for (size_t i{ 0ull }; i < fill; ++i)
    {
        extraNonceFill += '0';
    }

    // Define the gap each nonce for multi devices.
    std::string gapExtraNonce;
    for (auto i = 0ul; i < fill; ++i)
    {
        gapExtraNonce += 'f';
    }

    // Set nonce from extraNonce.
    jobInfo.gapNonce = std::strtoull(gapExtraNonce.c_str(), nullptr, 16);
    jobInfo.startNonce = std::strtoull(extraNonceFill.c_str(), nullptr, 16);
    jobInfo.nonce = jobInfo.startNonce;
}


void stratum::Stratum::setExtraNonce(
    std::string const& paramExtraNonce,
    uint32_t const paramExtraNonce2Size)
{
    setExtraNonce(paramExtraNonce);
    jobInfo.extraNonce2Size = paramExtraNonce2Size;
}


void stratum::Stratum::updateJob()
{
    if (false == isValidJob())
    {
        return;
    }
    if (nullptr != dispatchJob)
    {
        dispatchJob(uuid, jobInfo);
    }
    else
    {
        logErr() << "The callback updateJob is nullptr!";
    }
}


bool stratum::Stratum::isValidJob() const
{
    if (true == algo::isHashEmpty(jobInfo.jobID))
    {
        logDebug() << "jobID is empty";
        return false;
    }
    if (true == algo::isHashEmpty(jobInfo.headerHash))
    {
        logDebug() << "HeaderHash is empty";
        return false;
    }
    if (true == algo::isHashEmpty(jobInfo.boundary))
    {
        logDebug() << "BoundaryHash is empty";
        return false;
    }
    if (0ull == jobInfo.boundaryU64)
    {
        logDebug() << "Boundary U64 == 0ull";
        return false;
    }
    if (0 >= jobInfo.epoch)
    {
        logDebug() << "epoch == 0";
        return false;
    }
    if (0 >= jobInfo.nonce)
    {
        logDebug() << "nonce == 0";
        return false;
    }

    return true;
}


void stratum::Stratum::onShare(
    boost::json::object const& root,
    uint32_t const miningRequestID)
{
    if (stratum::Stratum::ID_MINING_SUBMIT <= miningRequestID)
    {
        ++shareTotal;
        if (   (false == common::boostJsonContains(root, "result") || false == root.at("result").as_bool())
            || (true == common::boostJsonContains(root, "error") && false == root.at("error").is_null()))
        {
            logErr() << root;
            ++shareInvalid;
        }
        else
        {
            ++shareValid;
        }
    }

    logInfo() << "Info SHARE:"
        << " valid[" << shareValid << "]"
        << " invalid[" << shareInvalid << "]"
        << " total[" << shareTotal << "]";
}
