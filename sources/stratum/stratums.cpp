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
            stratum = std::make_shared<stratum::StratumSha256>();
            break;
        }
        case algo::ALGORITHM::ETHASH:
        {
            stratum = std::make_shared<stratum::StratumEthash>();
            break;
        }
        case algo::ALGORITHM::ETCHASH:
        {
            stratum = std::make_shared<stratum::StratumEtchash>();
            break;
        }
        case algo::ALGORITHM::PROGPOW:
        {
            stratum = std::make_shared<stratum::StratumProgPOW>();
            break;
        }
        case algo::ALGORITHM::PROGPOWQUAI:
        {
            stratum = std::make_shared<stratum::StratumProgpowQuai>();
            break;
        }
        case algo::ALGORITHM::PROGPOWZ:
        {
            stratum = std::make_shared<stratum::StratumProgpowZ>();
            break;
        }
        case algo::ALGORITHM::KAWPOW:
        {
            stratum = std::make_shared<stratum::StratumKawPOW>();
            break;
        }
        case algo::ALGORITHM::MEOWPOW:
        {
            stratum = std::make_shared<stratum::StratumMeowPOW>();
            break;
        }
        case algo::ALGORITHM::FIROPOW:
        {
            stratum = std::make_shared<stratum::StratumFiroPOW>();
            break;
        }
        case algo::ALGORITHM::EVRPROGPOW:
        {
            stratum = std::make_shared<stratum::StratumEvrprogPOW>();
            break;
        }
        case algo::ALGORITHM::AUTOLYKOS_V2:
        {
            stratum = std::make_shared<stratum::StratumAutolykosV2>();
            break;
        }
        case algo::ALGORITHM::BLAKE3:
        {
            stratum = std::make_shared<stratum::StratumBlake3>();
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
