#include <algo/hash_utils.hpp>
#include <common/boost_utils.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/stratums.hpp>


void stratum::StratumSmartMining::setCallbackSetAlgorithm(callbackSetAlgorithm callback)
{
    doSetAlgorithm = callback;
}


void stratum::StratumSmartMining::setCallbackUpdateJob(callbackUpdateJob callback)
{
    doUpdateJob = callback;
}


void stratum::StratumSmartMining::setCallbackShareStatus(callbackShareStatus callback)
{
    doShareStatus = callback;
}


void stratum::StratumSmartMining::onConnect()
{
    subscribe();
}


void stratum::StratumSmartMining::onReceive(
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


void stratum::StratumSmartMining::onMethod(
    boost::json::object const& root)
{
    bool success { true };
    std::string const method{ root.at("method").as_string().c_str() };

    if ("smart_mining.set_algo" == method)
    {
        success = onSmartMiningSetAlgo(root);
    }
    else if ("smart_mining.set_extra_nonce" == method)
    {
        success = onSmartMiningSetExtraNonce(root);
    }
    else if ("mining.notify" == method)
    {
        success = onMiningNotify(root);
    }
    else if ("mining.set_difficulty" == method)
    {
        success = onMiningSetDifficulty(root);
    }
    else if ("mining.set_target" == method)
    {
        success = onMiningSetTarget(root);
    }
    else
    {
        success = false;
        logErr() << "Method: " << method << " unimplemented!";
    }

    if (false == success)
    {
        logErr() << "Error on method => " << root;
    }
}


bool stratum::StratumSmartMining::onSmartMiningSetAlgo(
    boost::json::object const& root)
{
    std::string const& algorithm { root.at("params").as_string().c_str() };
    currentAlgorithm = algo::toEnum(algorithm);
    doSetAlgorithm(currentAlgorithm);

    SAFE_DELETE(stratumPool);
    stratumPool = stratum::NewStratum(currentAlgorithm);
    IS_NULL(stratumPool);

    stratumPool->socketTCP = socketTCP;

    return true;
}


bool stratum::StratumSmartMining::onSmartMiningSetExtraNonce(
    boost::json::object const& root)
{
    IS_NULL(stratumPool);

    switch (currentAlgorithm)
    {
        case algo::ALGORITHM::SHA256:
        case algo::ALGORITHM::ETHASH:
        case algo::ALGORITHM::ETCHASH:
        case algo::ALGORITHM::PROGPOW:
        case algo::ALGORITHM::KAWPOW:
        case algo::ALGORITHM::MEOWPOW:
        case algo::ALGORITHM::FIROPOW:
        case algo::ALGORITHM::EVRPROGPOW:
        {
            std::string extraNonceStr { root.at("params").as_string().c_str() };
            stratumPool->setExtraNonce(extraNonceStr);
            break;
        }
        case algo::ALGORITHM::AUTOLYKOS_V2:
        {
            auto params { root.at("params").as_array() };
            std::string extraNonceStr { params.at(0).as_string().c_str() };
            uint32_t const extraNonce2Size { common::boostJsonGetNumber<uint32_t const>(params.at(1)) };
            stratumPool->setExtraNonce(extraNonceStr, extraNonce2Size);
            break;
        }
        default:
        {
            logErr() << "Unknow algorithm!";
            return false;
        }
    }

    return true;
}


bool stratum::StratumSmartMining::onMiningNotify(
    boost::json::object const& root)
{
    IS_NULL(stratumPool);

    stratumPool->onMiningNotify(root);
    doUpdateJob(stratumPool->jobInfo);

    return true;
}


bool stratum::StratumSmartMining::onMiningSetDifficulty(
    boost::json::object const& root)
{
    IS_NULL(stratumPool);

    stratumPool->onMiningSetDifficulty(root);

    return true;
}


bool stratum::StratumSmartMining::onMiningSetTarget(
    boost::json::object const& root)
{
    IS_NULL(stratumPool);

    stratumPool->onMiningSetTarget(root);

    return true;
}


void stratum::StratumSmartMining::onResponse(
    boost::json::object const& root)
{
    auto miningRequestID{ common::boostJsonGetNumber<uint32_t const>(root.at("id")) };

    if (miningRequestID <= 2)
    {
        return;
    }

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

    if (nullptr != doShareStatus)
    {
        doShareStatus(isValid, miningRequestID, 0u);
    }
}


void stratum::StratumSmartMining::subscribe()
{
    auto const& config { common::Config::instance() };
    boost::json::array listCoin{};

    for (auto const& [coin, poolConfig] : config.smartMining.coinPoolConfig)
    {
        boost::json::array array
        {
            coin,
            poolConfig.host,
            poolConfig.port,
            poolConfig.wallet
        };
        listCoin.emplace_back(array);
    }

    boost::json::object root;
    root["id"] = stratum::StratumSmartMining::ID_MINING_SUBSCRIBE;
    root["method"] = "mining.subscribe";
    root["params"] = boost::json::array
    {
        workerName,
        password,
        listCoin
    };

    send(root);
}


void stratum::StratumSmartMining::miningSubmit(
    uint32_t const deviceId,
    boost::json::array const& params)
{
    stratumPool->miningSubmit(deviceId, params);
}


void stratum::StratumSmartMining::miningSubmit(
    uint32_t const deviceId,
    boost::json::object const& params)
{
    stratumPool->miningSubmit(deviceId, params);
}
