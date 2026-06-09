#include <boost/asio/ssl/host_name_verification.hpp>

#include <common/log/log.hpp>
#include <network/network.hpp>

#if defined(_WIN32)
#pragma comment(lib, "crypt32.lib")
#include <string>

#include <wincrypt.h>


namespace
{
    // Validate the peer-presented certificate chain with the native Windows chain engine.
    // Unlike OpenSSL, this supplies the system trust anchors, fetches any intermediate the
    // server omitted via the certificate's AIA extension, and matches the hostname. Returns
    // true only when the chain is trusted AND valid for `host`.
    bool verifyChainWithWindows(X509_STORE_CTX* ctx, std::string const& host)
    {
        STACK_OF(X509)* const peerCerts{ X509_STORE_CTX_get0_untrusted(ctx) };
        X509* const           leaf{ X509_STORE_CTX_get0_cert(ctx) };
        if (nullptr == leaf)
        {
            logErr() << "TLS certificate is missing.";
            return false;
        }

        // In-memory store holding the certs the server actually sent, so the chain engine can
        // use any intermediates it did provide (and AIA-fetch the ones it did not).
        HCERTSTORE const winStore{
            CertOpenStore(CERT_STORE_PROV_MEMORY, X509_ASN_ENCODING, 0, CERT_STORE_CREATE_NEW_FLAG, nullptr)
        };
        if (nullptr == winStore)
        {
            logErr() << "Cannot allocate the Windows certificate store.";
            return false;
        }

        auto addCert{ [&winStore](X509* x509) -> PCCERT_CONTEXT
        {
            unsigned char* der{ nullptr };
            int const      len{ i2d_X509(x509, &der) };
            PCCERT_CONTEXT context{ nullptr };
            if (0 < len)
            {
                CertAddEncodedCertificateToStore(
                    winStore, X509_ASN_ENCODING, der, static_cast<DWORD>(len), CERT_STORE_ADD_ALWAYS, &context);
            }
            if (nullptr != der)
            {
                OPENSSL_free(der);
            }
            return context;
        } };

        int const peerCount{ nullptr != peerCerts ? sk_X509_num(peerCerts) : 0 };
        for (int i{ 0 }; i < peerCount; ++i)
        {
            if (PCCERT_CONTEXT const c{ addCert(sk_X509_value(peerCerts, i)) }; nullptr != c)
            {
                CertFreeCertificateContext(c);
            }
        }

        PCCERT_CONTEXT const leafContext{ addCert(leaf) };
        if (nullptr == leafContext)
        {
            logErr() << "Cannot encode the TLS leaf certificate.";
            CertCloseStore(winStore, 0);
            return false;
        }

        LPSTR           serverAuth[]{ const_cast<LPSTR>(szOID_PKIX_KP_SERVER_AUTH) };
        CERT_CHAIN_PARA chainPara{};
        chainPara.cbSize                                    = sizeof(chainPara);
        chainPara.RequestedUsage.dwType                     = USAGE_MATCH_TYPE_AND;
        chainPara.RequestedUsage.Usage.cUsageIdentifier     = 1;
        chainPara.RequestedUsage.Usage.rgpszUsageIdentifier = serverAuth;

        // hChainEngine = nullptr -> default engine (system roots + AIA fetch of a missing
        // intermediate). No revocation flags -> no CRL/OCSP network dependency.
        PCCERT_CHAIN_CONTEXT chainContext{ nullptr };
        BOOL const           built{ CertGetCertificateChain(
            nullptr, leafContext, nullptr, winStore, &chainPara, CERT_CHAIN_CACHE_END_CERT, nullptr, &chainContext) };

        bool verified{ false };
        if (FALSE != built && nullptr != chainContext)
        {
            std::wstring wideHost{ host.begin(), host.end() };

            SSL_EXTRA_CERT_CHAIN_POLICY_PARA sslPara{};
            sslPara.cbSize         = sizeof(sslPara);
            sslPara.dwAuthType     = AUTHTYPE_SERVER;
            sslPara.pwszServerName = wideHost.empty() ? nullptr : wideHost.data();

            CERT_CHAIN_POLICY_PARA policyPara{};
            policyPara.cbSize            = sizeof(policyPara);
            policyPara.pvExtraPolicyPara = &sslPara;

            CERT_CHAIN_POLICY_STATUS policyStatus{};
            policyStatus.cbSize = sizeof(policyStatus);

            if (FALSE != CertVerifyCertificateChainPolicy(CERT_CHAIN_POLICY_SSL, chainContext, &policyPara, &policyStatus))
            {
                verified = (0 == policyStatus.dwError);
                if (false == verified)
                {
                    logErr() << "TLS certificate chain rejected by Windows for " << host << " (status "
                             << policyStatus.dwError << ").";
                }
            }
        }
        else
        {
            logErr() << "Windows could not build a certificate chain for " << host << " (error "
                     << static_cast<unsigned long>(GetLastError()) << ").";
        }

        if (nullptr != chainContext)
        {
            CertFreeCertificateChain(chainContext);
        }
        CertFreeCertificateContext(leafContext);
        CertCloseStore(winStore, 0);
        return verified;
    }
}
#endif


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


bool network::NetworkTCPClient::onVerifySSL(bool preverified, boost_verify_context& ctx)
{
#if defined(_WIN32)
    // OpenSSL has no trust anchors on Windows (see doSecureConnection), so `preverified`
    // is always false here. The real trust decision is delegated to the Windows chain
    // engine, made once at the leaf (depth 0); let the higher chain levels pass so OpenSSL
    // keeps walking down to the leaf.
    (void) preverified;
    if (0 != X509_STORE_CTX_get_error_depth(ctx.native_handle()))
    {
        return true;
    }
    return verifyChainWithWindows(ctx.native_handle(), host);
#else
    // OpenSSL's chain / expiry / CA verification must pass first. The previous
    // implementation discarded `preverified` and returned true for any cert that
    // merely existed, which fully bypassed TLS authentication and let a MITM
    // present any certificate. Never override a chain failure.
    if (false == preverified)
    {
        logErr() << "TLS certificate chain verification failed.";
        return false;
    }

    auto const* cert{ X509_STORE_CTX_get_current_cert(ctx.native_handle()) };
    if (nullptr == cert)
    {
        logErr() << "TLS certificate is missing.";
        return false;
    }

    // Ensure the leaf certificate actually matches the pool hostname; a valid
    // certificate for a different host must not be accepted.
    boost::asio::ssl::host_name_verification const verifier{ host };
    if (false == verifier(preverified, ctx))
    {
        logErr() << "TLS certificate hostname mismatch for " << host << ".";
        return false;
    }

    return true;
#endif
}


void network::NetworkTCPClient::onReceiveAsync(boost_error_code const& ec, size_t bytes)
{
    try
    {
        if (boost_error::success == ec)
        {
            std::string msg;
            msg.assign(
                boost::asio::buffers_begin(recvBuffer.data()),
                boost::asio::buffers_begin(recvBuffer.data()) + bytes);

            recvBuffer.consume(bytes);

            if (false == msg.empty())
            {
                onReceive(msg);
            }
            asyncReceive();
        }
        else
        {
            if (boost::asio::error::eof == ec || boost::asio::error::connection_aborted == ec
                || boost::asio::error::connection_reset == ec || boost::asio::error::connection_refused == ec)
            {
                retryConnect();
            }
            else
            {
                logErr() << "Error during receiving message: "
                         << "[" << ec.value() << "] " << ec.message();
                disconnect();
            }
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


void network::NetworkTCPClient::onSend(boost_error_code const& ec, [[maybe_unused]] size_t const bytes)
{
    if (boost::system::errc::errc_t::success != ec)
    {
        logErr() << "Fail on send message to " << host << ":" << port;
    }
}