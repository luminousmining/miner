#include <vector>

#include <boost/json.hpp>
#include <boost/json/src.hpp>

#include <api/api.hpp>
#include <common/app.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>


void api::ServerAPI::setDeviceManager(device::DeviceManager* _deviceManager)
{
    deviceManager = _deviceManager;
}


void api::ServerAPI::setPort(
    uint32_t const _port)
{
    port = _port;
}


bool api::ServerAPI::bind()
{
    alive.store(true, boost::memory_order::seq_cst);

    threadDoAccept.interrupt();
    threadDoAccept = boost_thread{ boost::bind(&api::ServerAPI::loopAccept, this) };

    return true;
}


void api::ServerAPI::loopAccept()
{
    namespace boost_http = boost::beast::http;

    boost_io_context ioContext{};
    boost_acceptor acceptor{ ioContext, { boost_tcp::v4(), castU16(port) } };

    while (true == alive.load(boost::memory_order::relaxed))
    {
        ////////////////////////////////////////////////////////////////////////
        boost_socket socket{ ioContext };
        acceptor.accept(socket);

        ////////////////////////////////////////////////////////////////////////
        boost::beast::flat_buffer buffer{};
        boost_http::request<boost_http::string_body> request{};
        boost_http::read(socket, buffer, request);

        ////////////////////////////////////////////////////////////////////////
        onMessage(socket, request);

        ////////////////////////////////////////////////////////////////////////
        socket.shutdown(boost_tcp::socket::shutdown_send);
    }
}


void api::ServerAPI::onMessage(
    boost_socket& socket,
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
    else
    {
        response.result(boost_http::status::not_found);
    }
}


void api::ServerAPI::onHiveOSGetStats(
    boost_socket& socket,
    boost_response& response)
{
    ////////////////////////////////////////////////////////////////////////////
    std::string version
    {
        std::to_string(common::VERSION_MAJOR)
        + "."
        + std::to_string(common::VERSION_MINOR)
    };

    ////////////////////////////////////////////////////////////////////////////
    boost::json::object root
    {
        { "hs", boost::json::array{} },         // array of hashes
        { "hs_units", "hs" },                   // Optional: units that are uses for hashes array, "hs", "khs", "mhs", ... Default "khs".
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
    uint64_t sharesValid { 0ull };
    uint64_t sharesInvalid { 0ull };
    std::string sharesInvalidGpus{};

    ////////////////////////////////////////////////////////////////////////////
    std::vector<device::Device*> devices{ deviceManager->getDevices() };
    for (device::Device* device : devices)
    {
        sharesInvalidGpus += "0;";
        if (nullptr == device)
        {
            hs.push_back(0);
        }
        else
        {
            hs.push_back(castU64(device->getHashrate()));
            root["algo"] = algo::toString(device->algorithm);

            statistical::Statistical::ShareInfo shareInfo { device->getShare() };
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


void api::ServerAPI::onHiveOSGetTotalHashrate(
    boost_socket& socket,
    boost_response& response)
{
    uint64_t totalHashrate{ 0ull };
    boost::json::object root{};
    std::vector<device::Device*> devices{ deviceManager->getDevices() };

    for (device::Device* device : devices)
    {
        if (nullptr == device)
        {
            continue;
        }
        totalHashrate += castU64(device->getHashrate());
    }

    root["total_hash_rate"] = totalHashrate;

    response.body() = boost::json::serialize(root);
    response.prepare_payload();

    boost::beast::http::write(socket, response);
}
