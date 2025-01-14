#include <common/app.hpp>
#include <common/boost_utils.hpp>
#include <common/cast.hpp>
#include <common/config.hpp>
#include <algo/hash_utils.hpp>
#include <common/log/log.hpp>
#include <stratum/stratum.hpp>


stratum::Stratum::~Stratum()
{
}


void stratum::Stratum::miningSubmit(
    [[maybe_unused]] uint32_t const deviceId,
    [[maybe_unused]] boost::json::array const& params)
{
    logErr() << "mining.submit params[array] was not implemented!";
}


void stratum::Stratum::miningSubmit(
    [[maybe_unused]] uint32_t const deviceId,
    [[maybe_unused]] boost::json::object const& params)
{
    logErr() << "mining.submit params[object] was not implemented!";
}


void stratum::Stratum::setCallbackUpdateJob(
    stratum::Stratum::callbackUpdateJob cbUpdateJob)
{
    if (nullptr != cbUpdateJob)
    {
        dispatchJob = cbUpdateJob;
    }
}


void stratum::Stratum::setCallbackShareStatus(
    stratum::Stratum::callbackShareStatus cbShareStatus)
{
    if (nullptr != cbShareStatus)
    {
        shareStatus = cbShareStatus;
    }
}


void stratum::Stratum::onReceive(
    std::string const& message)
{
    try
    {
        auto root{ boost::json::parse(message).as_object() };
        logDebug() << "<--" << root;

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

    if ("mining.set" == method)
    {
        onMiningSet(root);
    }
    else if ("mining.notify" == method)
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
        boost::json::array const& params(root.at("params").as_array());
        std::stringstream ss;
        for (auto msg : params)
        {
            ss << msg;
        }
        logWarn() << "Pool: " << ss.str();
    }
    else
    {
        onUnknowMethod(root);
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

    switch(stratumType)
    {
        case stratum::STRATUM_TYPE::STRATUM:
        {
            miningSubscribe();
            break;
        }
        case stratum::STRATUM_TYPE::ETHEREUM_V2:
        {
            miningHello();
            break;
        }
    }
}


void stratum::Stratum::loopTimeout()
{
    boost::chrono::seconds const sec{ timeout - 5 };
    while (true == authenticated)
    {
        boost::this_thread::sleep_for(sec);

        boost::json::object root;
        root["id"] = stratum::Stratum::ID_MINING_SUBSCRIBE;
        root["method"] = "mining.noop";
        root["params"] = {};

        send(root);
    }
}


void stratum::Stratum::onUnknowMethod(
    boost::json::object const& root)
{
    std::string const method{ root.at("method").as_string().c_str() };

    logErr()
        << "Unknow[" << method << "]"
        << " " << root;
}


void stratum::Stratum::onMiningSet(
    [[maybe_unused]] boost::json::object const& root)
{
    logErr() << "mining.set unimplemented!";
}


void stratum::Stratum::onMiningSetTarget(
    [[maybe_unused]] boost::json::object const& root)
{
    logErr() << "mining.set_target unimplemented!";
}


void stratum::Stratum::onMiningSetExtraNonce(
    [[maybe_unused]] boost::json::object const& root)
{
    logErr() << "mining.set_extranonce unimplemented!";
}


void stratum::Stratum::doLoopTimeout()
{
    threadTimeout.interrupt();
    threadTimeout = boost_thread{ boost::bind(&stratum::Stratum::loopTimeout, this) };
}

void stratum::Stratum::miningHello()
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
    root["method"] = "mining.hello";
    root["params"] =
    {
        { "agent", softwareName },
        { "host", host },
        { "port", port },
        { "proto", stratumName }
    };

    send(root);
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

    if (stratumType == stratum::STRATUM_TYPE::STRATUM)
    {
        root["params"] = boost::json::array{ softwareName, stratumName };
    }

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
        common::Config const& config { common::Config::instance() };
        if (common::PROFILE::STANDARD == config.profile)
        {
            logErr() << "The callback updateJob is nullptr!";
        }
    }
}


bool stratum::Stratum::isValidJob() const
{
    bool ret { true };

    if (true == algo::isHashEmpty(jobInfo.jobID))
    {
        logDebug() << "jobID is empty";
        ret = false;
    }
    if (   true == algo::isHashEmpty(jobInfo.headerHash)
        && true == algo::isHashEmpty(jobInfo.headerBlob))
    {
        logDebug() << "HeaderHash or headerBlob is empty";
        ret = false;
    }
    if (true == algo::isHashEmpty(jobInfo.boundary))
    {
        logDebug() << "BoundaryHash is empty";
        ret = false;
    }
    if (0ull == jobInfo.boundaryU64)
    {
        logDebug() << "Boundary U64 == 0ull";
        ret = false;
    }
    if (0ull >= jobInfo.nonce)
    {
        logDebug() << "nonce == 0";
        ret = false;
    }

    return ret;
}


void stratum::Stratum::onShare(
    boost::json::object const& root,
    uint32_t const miningRequestID)
{
    bool isValid { true };
    bool const isErrResult
    {
           false == common::boostJsonContains(root, "result")
        || true == root.at("result").is_null()
        || false == root.at("result").as_bool()
    };
    bool const isErrError
    {
           true == common::boostJsonContains(root, "error")
        && false == root.at("error").is_null()
    };
    if (true == isErrResult || true == isErrError)
    {
        logErr() << root;
        isValid = false;
    }

    if (nullptr != shareStatus)
    {
        shareStatus(isValid, miningRequestID, uuid);
    }
}
