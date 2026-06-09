#pragma once

#include <memory>

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/json.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/thread.hpp>

#include <network/write_pump.hpp>


namespace network
{
    struct NetworkTCPClient : public std::enable_shared_from_this<NetworkTCPClient>
    {
      public:
        // boost alias
        using boost_socket = boost::asio::ssl::stream<boost::asio::ip::tcp::socket>;
        using boost_error = boost::system::errc::errc_t;
        using boost_resolve_flags = boost::asio::ip::resolver_base::flags;
        using boost_context = boost::asio::ssl::context;
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

        // Upper bound for a single '\n'-delimited stratum line. async_read_until
        // grows recvBuffer without limit otherwise, so a pool (or a MITM) that
        // streams bytes with no delimiter can exhaust memory. 64 KiB is far above
        // any real stratum message; an oversize line completes with an error and
        // the connection is dropped (fail-closed).
        static constexpr std::size_t MAX_RECV_STREAMBUF{ 64u * 1024u };

        // MAX RETRY CONNECTION
        static constexpr uint32_t MAX_RETRY_COUNT{ 10u };

        bool                    secureConnection{ false };
        std::string             host{};
        uint32_t                port{ 0u };
        uint32_t                countRetryConnect{ 0u };
        boost::asio::streambuf  recvBuffer{ MAX_RECV_STREAMBUF };
        boost_mutex             rxMutex;
        boost_thread            runService;
        boost::asio::io_context ioContext;
        boost_context           context{ boost_context::tlsv12_client };
        boost_socket*           socketTCP{ nullptr };

        // Serializes all outbound writes on socketTCP so only one async_write is
        // ever in flight. Created once in connect(); a SmartMining child shares
        // its parent's pump (single writer per shared socket).
        std::shared_ptr<network::WritePump> pump{ nullptr };

        NetworkTCPClient() = default;
        virtual ~NetworkTCPClient();

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
        void transmit(std::shared_ptr<std::string const> const& payload);
    };
}
