#if defined(_WIN32)
#pragma comment(lib, "crypt32.lib")
#endif


#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <common/system.hpp>
#include <network/network.hpp>


#include <boost/asio/buffer.hpp>
#include <boost/bind/bind.hpp>
#include <boost/chrono.hpp>


#if defined(_WIN32)
#include <wincrypt.h>
#endif


network::NetworkTCPClient::NetworkTCPClient()
{
}


network::NetworkTCPClient::~NetworkTCPClient()
{
    disconnect();
}


void network::NetworkTCPClient::wait()
{
    runService.join();
}


bool network::NetworkTCPClient::connect()
{
    try
    {
        logInfo() << "Connection to " << host << ":" << port;

        boost_error_code ec{};
        auto const& config{ common::Config::instance() };
        auto const address{ boost::asio::ip::address::from_string(host, ec) };

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

        socketTCP.reset(new boost_socket(ioService, context));

        if (boost_error::success != ec)
        {
            boost_resolver resolver{ ioService };
            boost_query    query{ host, std::to_string(port) };
            auto           endpoints{ resolver.resolve(query, ec) };

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
    catch(boost::exception const& e)
    {
        logErr() << diagnostic_information(e);
        return false;
    }
    catch(std::exception const& e)
    {
        logErr() << e.what();
        return false;
    }

    runService.interrupt();
    runService = boost::thread{ boost::bind(&boost::asio::io_service::run, &ioService) };

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

        logErr()
            << "Retry connection to " << host << ":" << port
            << " in " << config.mining.retryConnectionCount << "s"
            << " [" << countRetryConnect << "/"
            << network::NetworkTCPClient::MAX_RETRY_COUNT << "]";

        std::this_thread::sleep_for(std::chrono::seconds(config.mining.retryConnectionCount));

        if (false == connect())
        {
            retryConnect();
        }
    }
    catch(boost::exception const& e)
    {
        disconnect();
        logErr() << diagnostic_information(e);
    }
    catch(std::exception const& e)
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
    catch(boost::exception const& e)
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
            boost::bind(
                &NetworkTCPClient::onVerifySSL,
                this,
                std::placeholders::_1,
                std::placeholders::_2));

#if defined(_WIN32)
        auto certStore{ CertOpenSystemStore(0, "ROOT") };
        if (certStore == nullptr)
        {
            logErr() << "Certifcat Store \"ROOT\" was not found !";
            return false;
        }

        auto* store{ X509_STORE_new() };
        PCCERT_CONTEXT certContext{ nullptr };
        while (nullptr != (certContext = CertEnumCertificatesInStore(certStore, certContext)))
        {
            auto* x509
            {
                d2i_X509
                (
                    nullptr,
                    const_cast<const unsigned char**>(&(certContext->pbCertEncoded)),
                    certContext->cbCertEncoded
                )
            };
            if (nullptr != x509)
            {
                X509_STORE_add_cert(store, x509);
                X509_free(x509);
            }
        }

        CertFreeCertificateContext(certContext);
        CertCloseStore(certStore, 0);
        SSL_CTX_set_cert_store(context.native_handle(), store);
#elif defined(__linux__)
        auto certPath{ common::getEnv("SSL_CERT_FILE") };
        try
        {
            context.load_verify_file
            (
                nullptr != certPath
                    ? certPath
                    : "/etc/ssl/certs/ca-certificates.crt"
            );
        }
        catch (...)
        {
            logErr()
                << "Failed to load ca certificates. Either the file"
                << " '/etc/ssl/certs/ca-certificates.crt' does not exist" << "\n"
                << "or the environment variable SSL_CERT_FILE is set to an invalid or"
                << " inaccessible file." << "\n"
                << "It is possible that certificate verification can fail.";
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
            logErr()
                << "\n"
                << "This can have multiple reasons:" << "\n"
                << "* Root certs are either not installed or not found" << "\n"
                << "* Pool uses a self-signed certificate" << "\n"
                << "* Pool hostname you're connecting to does not match"
                << " the CN registered for the certificate." << "\n"
#if !defined(_WIN32)
                << "Possible fixes:" << "\n"
                << "* Make sure the file '/etc/ssl/certs/ca-certificates.crt' exists and"
                << " is accessible" << "\n"
                << "* Export the correct path via 'export " << "\n"
                << "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt' to the correct"
                << " file" << "\n"
                << " On most systems you can install"
                << " the 'ca-certificates' package" << "\n"
                << " You can also get the latest file here: " << "\n"
                << "https://curl.haxx.se/docs/caextract.html" << "\n"
#endif
                ;
        }
        return false;
    }

    return true;
}


void network::NetworkTCPClient::send(
    char const* data,
    size_t size)
{
    UNIQUE_LOCK(txMutex);

    if (true == secureConnection)
    {
        boost::asio::async_write(
            *socketTCP,
            boost::asio::buffer(data, size),
            boost::bind(
                &NetworkTCPClient::onSend,
                this,
                boost::asio::placeholders::error,
                boost::asio::placeholders::bytes_transferred));
    }
    else
    {
        boost::asio::async_write(
            socketTCP->next_layer(),
            boost::asio::buffer(data, size),
            boost::bind(
                &NetworkTCPClient::onSend,
                this,
                boost::asio::placeholders::error,
                boost::asio::placeholders::bytes_transferred));
    }
}


void network::NetworkTCPClient::send(
    boost_json const& root)
{
    std::ostringstream oss;
    oss << root;
    std::string str{ oss.str() + "\n" };

    logDebug() << root;
    send(str.c_str(), str.size());
}
