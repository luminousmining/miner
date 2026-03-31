#include <iomanip>
#include <sstream>
#include <string>

#include <algo/cuckatoo/cuckatoo.hpp>
#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <common/app.hpp>
#include <common/boost_utils.hpp>
#include <common/cast.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/cuckatoo.hpp>


////////////////////////////////////////////////////////////////////////////
// Pool connection flow
//
//  onConnect()          ← base Stratum calls miningSubscribe()
//      └─► miningSubscribe()  → miningLogin()
//              └─► send login JSON
//
//  onReceive(message)   ← base Stratum dispatches:
//      ├─► onResponse()       for messages without "method"
//      │       ├─► id == ID_MINING_SUBSCRIBE → login ack → (pool sends jobs)
//      │       └─► id >= ID_MINING_SUBMIT    → share accept/reject
//      └─► onUnknownMethod()  for unknown method names
//              └─► "job" → onJob()
//
////////////////////////////////////////////////////////////////////////////


void stratum::StratumCuckatoo::miningSubscribe()
{
    miningLogin();
}


void stratum::StratumCuckatoo::miningLogin()
{
    using namespace std::string_literals;

    boost::json::object root;
    root["id"]      = stratum::Stratum::ID_MINING_SUBSCRIBE;
    root["jsonrpc"] = "2.0";
    root["method"]  = "login";
    root["params"]  = boost::json::object{
        { "login", wallet },
        { "pass",  password.empty() ? "x"s : password },
        { "agent", "LuminousMiner/"s
                   + std::to_string(common::VERSION_MAJOR)
                   + "."s
                   + std::to_string(common::VERSION_MINOR) }
    };

    send(root);
}


void stratum::StratumCuckatoo::onResponse(boost::json::object const& root)
{
    auto const miningRequestID{ common::boostJsonGetNumber<uint32_t>(root.at("id")) };

    switch (miningRequestID)
    {
        case stratum::Stratum::ID_MINING_SUBSCRIBE:
        {
            ////////////////////////////////////////////////////////////////////
            // Login response
            bool const ok{
                (false == root.contains("error") || true == root.at("error").is_null())
                && root.contains("result")
                && root.at("result").as_bool()
            };

            if (true == ok)
            {
                authenticated = true;
                logInfo() << "Successful login!";
            }
            else
            {
                logErr() << "Login failed : " << root;
            }
            break;
        }
        default:
        {
            ////////////////////////////////////////////////////////////////////
            // Share submit response
            onShare(root, miningRequestID);
            break;
        }
    }
}


void stratum::StratumCuckatoo::onUnknownMethod(boost::json::object const& root)
{
    std::string const method{ common::boostGetString(root, "method") };

    if ("job" == method)
    {
        onJob(root);
    }
    else if ("login" == method)
    {
        ////////////////////////////////////////////////////////////////////////
        // Some pools (e.g. 2miners) send the login response as a notification
        // that contains both "method":"login" and "result":"ok", which the base
        // class routes here instead of to onResponse().
        bool const hasNoError{
            false == root.contains("error") || true == root.at("error").is_null()
        };
        bool const resultOk{
            root.contains("result")
            && (
                (root.at("result").is_string()
                 && "ok" == std::string{ root.at("result").as_string().c_str() })
                || (root.at("result").is_bool() && true == root.at("result").as_bool())
            )
        };

        if (true == hasNoError && true == resultOk)
        {
            authenticated = true;
            logInfo() << "Successful login!";
        }
        else
        {
            logErr() << "Login failed : " << root;
        }
    }
    else
    {
        logWarn() << "Unhandled pool method: " << method;
    }
}


void stratum::StratumCuckatoo::onJob(boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    boost::json::object const& params{ root.at("params").as_object() };

    ////////////////////////////////////////////////////////////////////////////
    // Job metadata
    jobInfo.grinJobId   = common::boostJsonGetNumber<uint32_t>(params.at("job_id"));
    jobInfo.blockNumber = common::boostJsonGetNumber<uint64_t>(params.at("height"));

    ////////////////////////////////////////////////////////////////////////////
    // Build a non-zero jobID from blockNumber + grinJobId so that isValidJob()
    // never rejects the job (it rejects any all-zero jobID hash).
    // Using std::to_string(grinJobId) alone would produce "0" when job_id==0,
    // which algo::toHash256 converts to an all-zero hash.
    {
        std::ostringstream oss;
        oss << std::hex
            << std::setw(16) << std::setfill('0') << jobInfo.blockNumber
            << std::setw(8)  << std::setfill('0') << jobInfo.grinJobId;
        jobInfo.jobIDStr = oss.str();
    }
    jobInfo.jobID = algo::toHash256(jobInfo.jobIDStr);

    ////////////////////////////////////////////////////////////////////////////
    // Epoch counter: Cuckatoo32 has no DAG epoch; reuse the field as a
    // monotonically increasing counter so the device detects new jobs.
    if (-1 == jobInfo.epoch)
    {
        jobInfo.epoch = 1;
    }
    else
    {
        ++jobInfo.epoch;
    }

    ////////////////////////////////////////////////////////////////////////////
    // pre_pow: hex-encoded block header prefix stored in headerBlob.
    // The miner appends the nonce (8 bytes LE) before hashing.
    std::string const prePowHex{ common::boostGetString(params, "pre_pow") };
    jobInfo.prePowSize  = castU32(prePowHex.size() / 2u);
    jobInfo.headerBlob  = algo::toHash<algo::hash3072>(prePowHex, algo::HASH_SHIFT::RIGHT);

    ////////////////////////////////////////////////////////////////////////////
    // Difficulty → boundary
    // Grin sends a plain uint64 difficulty.  Convert to a 256-bit boundary:
    //   boundary = 2^256 / difficulty
    uint64_t const difficulty{
        common::boostJsonGetNumber<uint64_t>(params.at("difficulty"))
    };
    jobInfo.boundary    = algo::toHash256(static_cast<double>(difficulty));
    jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);

    ////////////////////////////////////////////////////////////////////////////
    // Nonce: start from 1 (not 0 — the base isValidJob() rejects nonce==0).
    // The device loop will increment this each kernel invocation, and
    // device.cpp offsets it further by (gapNonce * deviceId) so each GPU
    // searches a different nonce range.
    jobInfo.nonce = 1ull;

    ////////////////////////////////////////////////////////////////////////////
    updateJob();

    logInfo() << "New job  height=" << jobInfo.blockNumber
              << "  job_id=" << jobInfo.grinJobId
              << "  difficulty=" << difficulty;
}


void stratum::StratumCuckatoo::onMiningNotify([[maybe_unused]] boost::json::object const& root)
{
    // Grin does not use mining.notify — jobs arrive via onJob().
}


void stratum::StratumCuckatoo::onMiningSetDifficulty([[maybe_unused]] boost::json::object const& root)
{
    // Grin embeds difficulty inside the job notification — not as a separate message.
}


void stratum::StratumCuckatoo::miningSubmit(
    uint32_t const              deviceId,
    boost::json::object const&  params)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxSubmit);

    ////////////////////////////////////////////////////////////////////////////
    boost::json::object root;
    root["id"]      = (deviceId + 1u) * stratum::Stratum::OVERCOM_NONCE;
    root["jsonrpc"] = "2.0";
    root["method"]  = "submit";
    root["params"]  = boost::json::object{
        { "edge_bits", algo::cuckatoo::EDGE_BITS },
        { "height",    common::boostJsonGetNumber<uint64_t>(params.at("height")) },
        { "job_id",    common::boostJsonGetNumber<uint32_t>(params.at("job_id")) },
        { "nonce",     common::boostJsonGetNumber<uint64_t>(params.at("nonce")) },
        { "pow",       params.at("pow") }  // boost::json::array of 42 uint32
    };

    send(root);
}
