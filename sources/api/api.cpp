#include <vector>

#include <boost/json.hpp>
#include <boost/json/src.hpp>

#include <api/api.hpp>
#include <common/app.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <device/device_manager.hpp>


void api::ServerAPI::setPort(uint32_t const _port)
{
    port = _port;
}


bool api::ServerAPI::bind()
{
    logInfo() << "Start API on port " << port;

    alive.store(true, boost::memory_order::seq_cst);

    threadDoAccept.interrupt();
    threadDoAccept = boost_thread{ boost::bind(&api::ServerAPI::loopAccept, this) };

    return true;
}


void api::ServerAPI::loopAccept()
{
    namespace boost_http = boost::beast::http;

    boost_io_context ioContext{};
    boost_acceptor   acceptor{ ioContext };

    ////////////////////////////////////////////////////////////////////////////
    // Bind the acceptor defensively: the stats API is non-essential, so a bind
    // failure (port already in use, or reserved by the OS) must not terminate
    // the miner. Log and disable the API instead.
    try
    {
        boost_endpoint const endpoint{ boost_tcp::v4(), castU16(port) };
        acceptor.open(endpoint.protocol());
        acceptor.set_option(boost::asio::socket_base::reuse_address{ true });
        acceptor.bind(endpoint);
        acceptor.listen();
    }
    catch (boost::system::system_error const& exception)
    {
        logErr() << "Cannot start API on port " << port << ": " << exception.what()
                 << ". API disabled, mining continues. On Windows the port may sit "
                    "in a reserved range (check `netsh int ipv4 show excludedportrange tcp`); "
                    "pick another port with --api_port.";
        alive.store(false, boost::memory_order::seq_cst);
        return;
    }

    while (true == alive.load(boost::memory_order::relaxed))
    {
        ////////////////////////////////////////////////////////////////////////
        // A malformed or aborted client request must not bring down the miner
        // either, so handle each connection inside its own guard.
        try
        {
            boost_socket socket{ ioContext };
            acceptor.accept(socket);

            ////////////////////////////////////////////////////////////////////
            boost::beast::flat_buffer                    buffer{};
            boost_http::request<boost_http::string_body> request{};
            boost_http::read(socket, buffer, request);

            ////////////////////////////////////////////////////////////////////
            onMessage(socket, request);

            ////////////////////////////////////////////////////////////////////
            socket.shutdown(boost_tcp::socket::shutdown_send);
        }
        catch (boost::system::system_error const& exception)
        {
            logWarn() << "API request handling error: " << exception.what();
        }
    }
}


void api::ServerAPI::onMessage(
    boost_socket&                                                       socket,
    boost::beast::http::request<boost::beast::http::string_body> const& request)
{
    namespace boost_http = boost::beast::http;

    ////////////////////////////////////////////////////////////////////////
    boost_string_view target{ request.base().target() };

    ////////////////////////////////////////////////////////////////////////
    boost_http::response<boost_http::string_body> response{};
    response.version(request.version());
    response.set(boost_http::field::server, "LuminousMiner API");
    response.set(boost_http::field::content_type, "application/json");

    ////////////////////////////////////////////////////////////////////////
    if ("/hiveos/getStats" == target)
    {
        onHiveOSGetStats(socket, response);
        response.result(boost_http::status::ok);
    }
    else if ("/hiveos/getTotalHashrate" == target)
    {
        onHiveOSGetTotalHashrate(socket, response);
        response.result(boost_http::status::ok);
    }
    else if ("/api/get_stats" == target)
    {
        onWebGetStats(socket, response);
        response.result(boost_http::status::ok);
    }
    else
    {
        response.result(boost_http::status::not_found);
    }
}


void api::ServerAPI::onHiveOSGetStats(boost_socket& socket, boost_response& response)
{
    ////////////////////////////////////////////////////////////////////////////
    std::string version{ std::to_string(common::VERSION_MAJOR) + "." + std::to_string(common::VERSION_MINOR) };

    ////////////////////////////////////////////////////////////////////////////
    boost::json::object root{
        { "hs", boost::json::array{} }, // array of hashes
        { "hs_units", "hs" }, // Optional: units that are uses for hashes array, "hs", "khs", "mhs", ... Default "khs".
        { "temp", boost::json::array{} },       // array of miner temps
        { "fan", boost::json::array{} },        // array of miner fans
        { "uptime", 0 },                        // seconds elapsed from miner stats
        { "ver", version },                     // miner version currently run, parsed from it's api or manifest
        { "ar", boost::json::array{} },         // Optional: acceped, rejected shares
        { "algo", "" },                         // Optional: algo used by miner, should one of the exiting in Hive
        { "bus_numbers", boost::json::array{} } // Pci buses array in decimal format. E.g. 0a:00.0 is 10
    };
    boost::json::array hs{};
    boost::json::array temp{};
    boost::json::array fan{};
    boost::json::array ar{};
    boost::json::array busNumbers{};

    ////////////////////////////////////////////////////////////////////////////
    uint64_t    sharesValid{ 0ull };
    uint64_t    sharesInvalid{ 0ull };
    std::string sharesInvalidGpus{};

    ////////////////////////////////////////////////////////////////////////////
    auto&                        deviceManager{ device::DeviceManager::instance() };
    std::vector<device::Device*> devices{ deviceManager.getDevices() };
    for (device::Device* device : devices)
    {
        sharesInvalidGpus += "0;";
        if (nullptr == device) [[unlikely]]
        {
            hs.push_back(0);
        }
        else
        {
            hs.push_back(castU64(device->getHashrate()));
            root["algo"] = algo::toString(device->algorithm);

            statistical::Statistical::ShareInfo shareInfo{ device->getShare() };
            sharesInvalid += shareInfo.invalid;
            sharesValid += shareInfo.valid;
        }
        temp.push_back(0);
        fan.push_back(0);
        busNumbers.push_back(device->pciBus);
    }
    ar.push_back(sharesValid);               // share valid
    ar.push_back(sharesInvalid);             // share rejected
    ar.push_back(0);                         // share invalid
    ar.push_back(sharesInvalidGpus.c_str()); // shares invalid by gpus


    root["hs"] = hs;
    root["temp"] = temp;
    root["fan"] = fan;
    root["ar"] = ar;
    root["bus_numbers"] = busNumbers;

    ////////////////////////////////////////////////////////////////////////////
    response.body() = boost::json::serialize(root);
    response.prepare_payload();

    ////////////////////////////////////////////////////////////////////////////
    boost::beast::http::write(socket, response);
}


void api::ServerAPI::onHiveOSGetTotalHashrate(boost_socket& socket, boost_response& response)
{
    ////////////////////////////////////////////////////////////////////////////
    uint64_t                     totalHashrate{ 0ull };
    boost::json::object          root{};
    auto&                        deviceManager{ device::DeviceManager::instance() };
    std::vector<device::Device*> devices{ deviceManager.getDevices() };

    ////////////////////////////////////////////////////////////////////////////
    for (device::Device* device : devices)
    {
        if (nullptr == device) [[unlikely]]
        {
            continue;
        }
        totalHashrate += castU64(device->getHashrate());
    }

    ////////////////////////////////////////////////////////////////////////////
    root["total_hash_rate"] = totalHashrate;

    ////////////////////////////////////////////////////////////////////////////
    response.body() = boost::json::serialize(root);
    response.prepare_payload();

    ////////////////////////////////////////////////////////////////////////////
    boost::beast::http::write(socket, response);
}


void api::ServerAPI::onWebGetStats(boost_socket& socket, boost_response& response)
{
    ////////////////////////////////////////////////////////////////////////////
    std::string version{ std::to_string(common::VERSION_MAJOR) + "." + std::to_string(common::VERSION_MINOR) };

    ////////////////////////////////////////////////////////////////////////////
    boost::json::object root{
        { "hs", boost::json::array{} }, // Hashrates by device (GPU)
        { "hs_units", "hs" }, // Optional: units that are uses for hashes array, "hs", "khs", "mhs", ... Default "khs"
        { "temp", boost::json::array{} },   // Temperature by device (GPU)
        { "fan", boost::json::array{} },    // Fans speed by device (GPU)
        { "uptime", 0 },                    // Seconds elapsed from miner stats
        { "ver", version },                 // Miner version currently run
        { "shares", boost::json::array{} }, // Acceped and rejected shares
    };
    boost::json::array hs{};
    boost::json::array temp{};
    boost::json::array fan{};
    boost::json::array shares{};

    ////////////////////////////////////////////////////////////////////////////
    uint64_t    sharesValid{ 0ull };
    uint64_t    sharesInvalid{ 0ull };
    std::string sharesInvalidGpus{};

    ////////////////////////////////////////////////////////////////////////////
    auto&                        deviceManager{ device::DeviceManager::instance() };
    std::vector<device::Device*> devices{ deviceManager.getDevices() };
    for (device::Device* device : devices)
    {
        sharesInvalidGpus += "0;";
        if (nullptr == device) [[unlikely]]
        {
            hs.push_back(0);
        }
        else
        {
            hs.push_back(castU64(device->getHashrate()));

            statistical::Statistical::ShareInfo shareInfo{ device->getShare() };
            sharesInvalid += shareInfo.invalid;
            sharesValid += shareInfo.valid;
        }
        temp.push_back(0);
        fan.push_back(0);
    }
    shares.push_back(sharesValid);               // share valid
    shares.push_back(sharesInvalid);             // share rejected
    shares.push_back(0);                         // share invalid
    shares.push_back(sharesInvalidGpus.c_str()); // shares invalid by gpus

    root["hs"] = hs;
    root["temp"] = temp;
    root["fan"] = fan;
    root["shares"] = shares;

    ////////////////////////////////////////////////////////////////////////////
    response.body() = boost::json::serialize(root);
    response.prepare_payload();
    response.set("Access-Control-Allow-Origin", "*");

    ////////////////////////////////////////////////////////////////////////////
    boost::beast::http::write(socket, response);
}
