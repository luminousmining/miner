#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/stratums.hpp>


std::shared_ptr<stratum::Stratum> stratum::NewStratum(algo::ALGORITHM const algorithm)
{
    std::shared_ptr<stratum::Stratum> stratum{ nullptr };
    auto const&                       config{ common::Config::instance() };

    switch (algorithm)
    {
        case algo::ALGORITHM::SHA256:
        {
            stratum = NEW_SHARED(stratum::StratumSha256);
            break;
        }
        case algo::ALGORITHM::ETHASH:
        {
            stratum = NEW_SHARED(stratum::StratumEthash);
            break;
        }
        case algo::ALGORITHM::ETCHASH:
        {
            stratum = NEW_SHARED(stratum::StratumEtchash);
            break;
        }
        case algo::ALGORITHM::PROGPOW:
        {
            stratum = NEW_SHARED(stratum::StratumProgPOW);
            break;
        }
        case algo::ALGORITHM::PROGPOWQUAI:
        {
            stratum = NEW_SHARED(stratum::StratumProgpowQuai);
            break;
        }
        case algo::ALGORITHM::PROGPOWZ:
        {
            stratum = NEW_SHARED(stratum::StratumProgpowZ);
            break;
        }
        case algo::ALGORITHM::KAWPOW:
        {
            stratum = NEW_SHARED(stratum::StratumKawPOW);
            break;
        }
        case algo::ALGORITHM::MEOWPOW:
        {
            stratum = NEW_SHARED(stratum::StratumMeowPOW);
            break;
        }
        case algo::ALGORITHM::FIROPOW:
        {
            stratum = NEW_SHARED(stratum::StratumFiroPOW);
            break;
        }
        case algo::ALGORITHM::EVRPROGPOW:
        {
            stratum = NEW_SHARED(stratum::StratumEvrprogPOW);
            break;
        }
        case algo::ALGORITHM::AUTOLYKOS_V2:
        {
            stratum = NEW_SHARED(stratum::StratumAutolykosV2);
            break;
        }
        case algo::ALGORITHM::BLAKE3:
        {
            stratum = NEW_SHARED(stratum::StratumBlake3);
            break;
        }
        case algo::ALGORITHM::KHEAVYHASH:
        {
            stratum = NEW(stratum::StratumKHeavyHash);
            break;
        }
        case algo::ALGORITHM::UNKNOWN:
        {
            break;
        }
    }

    if (nullptr == stratum) [[unlikely]]
    {
        logErr() << "Fail alloc stratum for " << algorithm;
        return nullptr;
    }

    stratum->stratumType = config.mining.stratumType;

    switch (stratum->stratumType)
    {
        case stratum::STRATUM_TYPE::ETHEREUM_V1:
        {
            stratum->protocol = "EthereumStratum/1.0.0";
            break;
        }
        case stratum::STRATUM_TYPE::ETHEREUM_V2:
        {
            stratum->protocol = "EthereumStratum/2.0.0";
            break;
        }
        case stratum::STRATUM_TYPE::ETHPROXY:
        {
            stratum->protocol.clear();
            break;
        }
    }

    return stratum;
}
