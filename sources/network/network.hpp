#pragma once

#include <memory>


#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/json.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/thread.hpp>


namespace network
{
    struct NetworkTCPClient
    {
    public:
        // boost alias
        using boost_socket = boost::asio::ssl::stream<boost::asio::ip::tcp::socket>;
        using boost_error = boost::system::errc::errc_t;
        using boost_resolve_flags = boost::asio::ip::resolver_base::flags;
        using boost_context = boost::asio::ssl::context;
        using boost_queue = boost::lockfree::queue<std::string*>;
        using boost_resolver = boost::asio::ip::tcp::resolver;
        using boost_endpoint = boost::asio::ip::tcp::endpoint;
        using boost_mutex = boost::mutex;
        using boost_thread = boost::thread;
        using boost_error_code = boost::system::error_code;
        using boost_verify_context = boost::asio::ssl::verify_context;
        using boost_json = boost::json::object;

        // BUFFER SIZE MAX
        static constexpr uint32_t MAX_BUFFER_RECV{ 1024u };
        static constexpr uint32_t MAX_BUFFER_SEND{ 1024u };

        // MAX RETRY CONNECTION
        static constexpr uint32_t MAX_RETRY_COUNT{ 10u };

        bool                    secureConnection{ false };
        std::string             host{};
        uint32_t                port{ 0u };
        uint32_t                countRetryConnect{ 0u };
        boost::asio::streambuf  recvBuffer;
        boost_mutex             rxMutex;
        boost_mutex             txMutex;
        boost_thread            runService;
        boost::asio::io_context ioContext;
        boost_queue             tx{ 100 };
        boost_context           context{ boost_context::tlsv12_client };
        boost_socket*           socketTCP{ nullptr };

        NetworkTCPClient() = default;
        ~NetworkTCPClient() = default;

        virtual void onConnect() = 0;
        virtual void onReceive(std::string const& message) = 0;

        void wait();
        bool connect();
        void send(char const* data, std::size_t size);
        void send(boost::json::object const& object);

    private:
        void retryConnect();
        void shutdown();
        void disconnect();
        bool doSecureConnection();
        bool handshake();
        void asyncReceive();
        bool onVerifySSL(bool preverified, boost_verify_context& ctx);
        void onReceiveAsync(boost_error_code const& ec, size_t bytes);
        void onSend(boost_error_code const& ec, size_t bytes);
    };
}
