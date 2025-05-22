#pragma once

#include <string>

#include <boost/asio.hpp>
#include <boost/atomic/atomic.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

#include <device/device_manager.hpp>


namespace api
{
    struct ServerAPI
    {
    public:
        using boost_error_code = boost::beast::error_code;
        using boost_address = boost::asio::ip::address;
        using boost_io_context = boost::asio::io_context;
        using boost_tcp = boost::asio::ip::tcp;
        using boost_acceptor = boost::asio::ip::tcp::acceptor;
        using boost_socket = boost::asio::ip::tcp::socket;
        using boost_endpoint = boost::asio::ip::tcp::endpoint;
        using boost_thread = boost::thread;
        using boost_mutex = boost::mutex;
        using boost_atomic_bool = boost::atomic_bool;
        using boost_string_view = boost::beast::string_view;
        using boost_request = boost::beast::http::request<boost::beast::http::string_body>;
        using boost_response = boost::beast::http::response<boost::beast::http::string_body>;

        uint32_t        port{ 0u };
        boost_address   address{};

        boost_thread      threadDoAccept{};
        boost_mutex       mtx{};
        boost_atomic_bool alive { false };

        void setDeviceManager(device::DeviceManager* _deviceManager);
        void setPort(uint32_t const _port);
        bool bind();
        void loopAccept();

    private:
        device::DeviceManager* deviceManager{ nullptr };
        void onMessage(boost_socket& socket,
                       boost_request const& request);
        void onHiveOSGetStats(boost_socket& socket,
                              boost_response& response);
        void onHiveOSGetTotalHashrate(boost_socket& socket,
                                      boost_response& response);
        void onWebGetStats(boost_socket& socket,
                           boost_response& response);
    };
}
