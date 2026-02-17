#include <common/log/log.hpp>
#include <network/network.hpp>


void network::NetworkTCPClient::asyncReceive()
{
    if (true == secureConnection)
    {
        boost::asio::async_read_until(
            *socketTCP,
            recvBuffer,
            '\n',
            boost::bind(
                &NetworkTCPClient::onReceiveAsync,
                this,
                boost::asio::placeholders::error,
                boost::asio::placeholders::bytes_transferred));
    }
    else
    {
        boost::asio::async_read_until(
            socketTCP->next_layer(),
            recvBuffer,
            '\n',
            boost::bind(
                &NetworkTCPClient::onReceiveAsync,
                this,
                boost::asio::placeholders::error,
                boost::asio::placeholders::bytes_transferred));
    }
}


bool network::NetworkTCPClient::onVerifySSL(
    [[maybe_unused]] bool preverified,
    boost_verify_context& ctx)
{
    auto const* cert{ X509_STORE_CTX_get_current_cert(ctx.native_handle()) };
    if (nullptr == cert)
    {
        logErr() << "Certificat is incorrect.";
        return false;
    }

    return true;
}


void network::NetworkTCPClient::onReceiveAsync(
    boost_error_code const& ec,
    size_t bytes)
{
    try
    {
        if (boost_error::success == ec)
        {
            std::string msg;
            msg.assign(boost::asio::buffers_begin(recvBuffer.data()), boost::asio::buffers_begin(recvBuffer.data()) + bytes);

            recvBuffer.consume(bytes);

            if (false == msg.empty())
            {
                onReceive(msg);
            }
            asyncReceive();
        }
        else
        {
            if (   boost::asio::error::eof == ec
                || boost::asio::error::connection_aborted == ec
                || boost::asio::error::connection_reset == ec
                || boost::asio::error::connection_refused == ec)
            {
                retryConnect();
            }
            else
            {
                logErr()
                    << "Error during receiving message: "
                    << "[" << ec.value() << "] "
                    << ec.message();
                disconnect();
            }
        }
    }
    catch(boost::exception const& e)
    {
        logErr() << diagnostic_information(e);
    }
    catch(std::exception const& e)
    {
        logErr() << e.what();
    }
}


void network::NetworkTCPClient::onSend(
    boost_error_code const& ec,
    [[maybe_unused]] size_t const bytes)
{
    if (boost::system::errc::errc_t::success != ec)
    {
        logErr() << "Fail on send message to " << host << ":" << port;
    }
}