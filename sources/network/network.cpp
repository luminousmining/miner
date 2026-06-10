#include <boost/asio/buffer.hpp>
#include <boost/bind/bind.hpp>
#include <boost/chrono.hpp>

#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/env_utils.hpp>
#include <common/log/log.hpp>
#include <network/network.hpp>
#include <network/socks5.hpp>
#include <stratum/stratum_type.hpp>


network::NetworkTCPClient::~NetworkTCPClient()
{
    SAFE_DELETE(socketTCP);
}


void network::NetworkTCPClient::wait()
{
    runService.join();
}


bool network::NetworkTCPClient::connect()
{
    try
    {
        boost_error_code ec{};
        auto const&      config{ common::Config::instance() };

        logInfo() << "Connection to " << host << ":" << port << " with stratum "
                  << stratum::toString(config.mining.stratumType);

        if (true == config.mining.secrureConnect)
        {
            if (false == doSecureConnection())
            {
                return false;
            }
        }
        else
        {
            context.set_verify_mode(boost::asio::ssl::verify_none);
        }

        SAFE_DELETE(socketTCP);
        socketTCP = NEW(boost_socket(ioContext, context));
        boost_resolver resolver{ ioContext };
        if (true == config.mining.socks5)
        {
            if (std::nullopt != config.mining.socksHost)
            {
                auto const     addr{ boost::asio::ip::make_address(*config.mining.socksHost, ec) };
                boost_endpoint socksEndpoint{ addr, static_cast<boost::asio::ip::port_type>(config.mining.socksPort) };

                auto endpoints{ resolver.resolve(host, std::to_string(port), ec) };
                if (boost_error::success != ec || endpoints.begin() == endpoints.end())
                {
                    logErr() << "Cannot resolve " << host << ":" << port;
                    return false;
                }

                auto targetEndpoint{ *endpoints.begin() };
                socks5::proxy_connect(socketTCP->next_layer(), targetEndpoint, socksEndpoint, ec);

                if (socks5::result_code::ok != ec)
                {
                    logErr() << "Cannot connect to " << host << ":" << port << " with SOCKS5 proxy on "
                             << *config.mining.socksHost << ":" << config.mining.socksPort;
                    return false;
                }
            }
        }
        else
        {
            // from_string is no longer a memeber of boost::asio::ip::address in 1.90.
            auto const address{ boost::asio::ip::make_address(host, ec) };
            if (boost_error::success != ec)
            {
                auto endpoints{
                    resolver.resolve(host, std::to_string(port), boost_resolve_flags::numeric_service, ec)
                };

                if (boost_error::success != ec)
                {
                    logErr() << "Cannot resolve " << host << ":" << port;
                    return false;
                }

                boost::asio::connect(socketTCP->next_layer(), endpoints, ec);
                if (boost_error::success != ec)
                {
                    logErr() << "Cannot connect to DNS " << host << ":" << port;
                    return false;
                }
            }
            else
            {
                boost_endpoint endpoint{ address, static_cast<boost::asio::ip::port_type>(port) };
                socketTCP->next_layer().connect(endpoint, ec);
                if (boost_error::success != ec)
                {
                    logErr() << "Cannot connect to host " << host << ":" << port;
                    return false;
                }
            }
        }

        socketTCP->next_layer().set_option(boost::asio::socket_base::keep_alive(true));
        socketTCP->next_layer().set_option(boost::asio::ip::tcp::no_delay(true));

        if (true == config.mining.secrureConnect)
        {
            if (false == handshake())
            {
                return false;
            }
        }

        countRetryConnect = 0;
        onConnect();

        asyncReceive();
    }
    catch (boost::exception const& e)
    {
        logErr() << diagnostic_information(e);
        return false;
    }
    catch (std::exception const& e)
    {
        logErr() << e.what();
        return false;
    }

    runService.interrupt();
    runService = boost::thread{ boost::bind(&boost::asio::io_context::run, &ioContext) };

    return true;
}


void network::NetworkTCPClient::retryConnect()
{
    try
    {
        auto const& config{ common::Config::instance() };

        ++countRetryConnect;
        if (countRetryConnect > network::NetworkTCPClient::MAX_RETRY_COUNT)
        {
            disconnect();
            return;
        }
        disconnect();

        logErr() << "Retry connection to " << host << ":" << port << " in " << config.mining.retryConnectionCount << "s"
                 << " [" << countRetryConnect << "/" << network::NetworkTCPClient::MAX_RETRY_COUNT << "]";

        std::this_thread::sleep_for(std::chrono::seconds(config.mining.retryConnectionCount));

        if (false == connect())
        {
            retryConnect();
        }
    }
    catch (boost::exception const& e)
    {
        disconnect();
        logErr() << diagnostic_information(e);
    }
    catch (std::exception const& e)
    {
        disconnect();
        logErr() << e.what();
    }
}


void network::NetworkTCPClient::shutdown()
{
    disconnect();
    runService.interrupt();
}


void network::NetworkTCPClient::disconnect()
{
    try
    {
        if (nullptr != socketTCP)
        {
            logWarn() << "Disconnecting to " << host << ":" << port;
            if (true == socketTCP->next_layer().is_open())
            {
                socketTCP->next_layer().close();
            }
            else
            {
                socketTCP->next_layer().cancel();
            }
        }
    }
    catch (boost::exception const& e)
    {
        logErr() << diagnostic_information(e);
    }
}


bool network::NetworkTCPClient::doSecureConnection()
{
    try
    {
        secureConnection = true;

        context.set_verify_mode(boost::asio::ssl::verify_peer);
        context.set_verify_callback(
            boost::bind(&NetworkTCPClient::onVerifySSL, this, std::placeholders::_1, std::placeholders::_2));

#if defined(_WIN32)
        // No static trust store is seeded on Windows. OpenSSL cannot complete a chain whose
        // intermediate the server omits (it has no AIA fetching), a very common pool
        // misconfiguration. Instead onVerifySSL() hands the peer chain to the Windows chain
        // engine (CertGetCertificateChain), which supplies the system roots, AIA-fetches any
        // missing intermediate, and validates the hostname. verify_peer (set above) is still
        // required so the callback runs; the empty OpenSSL store just makes `preverified`
        // always false, which onVerifySSL expects on Windows.
#elif defined(__linux__)
        // Fall back to OpenSSL's built-in default trust paths if the explicit
        // bundle below is absent, so verification still succeeds (fail-closed).
        context.set_default_verify_paths();
        auto certPath{ common::getEnv("SSL_CERT_FILE") };
        try
        {
            context.load_verify_file(nullptr != certPath ? certPath : "/etc/ssl/certs/ca-certificates.crt");
        }
        catch (...)
        {
            logErr() << "Failed to load ca certificates. Either the file"
                     << " '/etc/ssl/certs/ca-certificates.crt' does not exist"
                     << "\n"
                     << "or the environment variable SSL_CERT_FILE is set to an invalid or"
                     << " inaccessible file."
                     << "\n"
                     << "It is possible that certificate verification can fail.";
        }
#elif defined(__APPLE__)
        // macOS: OpenSSL (vcpkg/Homebrew) is not wired into the system Keychain,
        // but ships its own CA bundle and honours SSL_CERT_FILE/SSL_CERT_DIR. Use
        // its default trust paths, and load an explicit bundle only when the user
        // points SSL_CERT_FILE at one.
        context.set_default_verify_paths();
        if (char const* const certPath{ common::getEnv("SSL_CERT_FILE") }; nullptr != certPath)
        {
            try
            {
                context.load_verify_file(certPath);
            }
            catch (...)
            {
                logErr() << "SSL_CERT_FILE is set but the file could not be loaded;"
                         << " certificate verification may fail.";
            }
        }
#endif
    }
    catch (boost::exception const& e)
    {
        logErr() << diagnostic_information(e);
        return false;
    }
    catch (std::exception const& e)
    {
        logErr() << e.what();
        return false;
    }

    return true;
}


bool network::NetworkTCPClient::handshake()
{
    boost_error_code ec{};

    socketTCP->handshake(boost::asio::ssl::stream_base::client, ec);

    if (boost_error::success != ec)
    {
        logErr() << "Cannot connect to pool with SSL option";
        if (337047686 == ec.value())
        {
            logErr() << "\n"
                     << "This can have multiple reasons:"
                     << "\n"
                     << "* Root certs are either not installed or not found"
                     << "\n"
                     << "* Pool uses a self-signed certificate"
                     << "\n"
                     << "* Pool hostname you're connecting to does not match"
                     << " the CN registered for the certificate."
                     << "\n"
#if !defined(_WIN32)
                     << "Possible fixes:"
                     << "\n"
                     << "* Make sure the file '/etc/ssl/certs/ca-certificates.crt' exists and"
                     << " is accessible"
                     << "\n"
                     << "* Export the correct path via 'export "
                     << "\n"
                     << "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt' to the correct"
                     << " file"
                     << "\n"
                     << " On most systems you can install"
                     << " the 'ca-certificates' package"
                     << "\n"
                     << " You can also get the latest file here: "
                     << "\n"
                     << "https://curl.haxx.se/docs/caextract.html"
                     << "\n"
#endif
                ;
        }
        return false;
    }

    return true;
}


void network::NetworkTCPClient::send(char const* data, size_t size)
{
    UNIQUE_LOCK(txMutex);

    if (nullptr == socketTCP) [[unlikely]]
    {
        logErr() << "Cannot send packet, socketTCP is nullptr!";
        return;
    }

    // async_write does NOT copy the buffer it is given: the storage must stay
    // valid until the operation completes. Callers pass pointers to temporaries
    // (e.g. send(boost_json) hands over a local string's c_str()), so the data
    // was being freed before the write finished -- a use-after-free. Own a copy
    // for the lifetime of the async op by capturing it in the completion handler.
    // shared_ptr (not unique_ptr) keeps the handler copyable: async_write is a
    // composed operation that may copy the handler through its internal layers,
    // and a captured unique_ptr would make the lambda move-only.
    auto payload{ std::make_shared<std::string>(data, size) };
    auto handler{ [this, payload](boost_error_code const& ec, std::size_t bytes)
                  { onSend(ec, bytes); } };

    if (true == secureConnection)
    {
        boost::asio::async_write(*socketTCP, boost::asio::buffer(*payload), std::move(handler));
    }
    else
    {
        boost::asio::async_write(socketTCP->next_layer(), boost::asio::buffer(*payload), std::move(handler));
    }
}


void network::NetworkTCPClient::send(boost_json const& root)
{
    std::ostringstream oss;
    oss << root;
    std::string str{ oss.str() + "\n" };

    logDebug() << "-->" << root;
    send(str.c_str(), str.size());
}
