#include <common/custom.hpp>
#include <common/config.hpp>
#include <common/log/log.hpp>
#include <stratum/stratums.hpp>


stratum::Stratum* stratum::NewStratum(
    algo::ALGORITHM const algorithm)
{
    stratum::Stratum* stratum { nullptr };
    auto const& config{ common::Config::instance() };

    switch (algorithm)
    {
        case algo::ALGORITHM::SHA256:
        {
            stratum = NEW(stratum::StratumSha256);
            break;
        }
        case algo::ALGORITHM::ETHASH:
        {
            stratum = NEW(stratum::StratumEthash);
            break;
        }
        case algo::ALGORITHM::ETCHASH:
        {
            stratum = NEW(stratum::StratumEtchash);
            break;
        }
        case algo::ALGORITHM::PROGPOW:
        {
            stratum = NEW(stratum::StratumProgPOW);
            break;
        }
        case algo::ALGORITHM::PROGPOWZ:
        {
            stratum = NEW(stratum::StratumProgpowZ);
            break;
        }
        case algo::ALGORITHM::PROGPOWQUAI:
        case algo::ALGORITHM::KAWPOW:
        {
            stratum = NEW(stratum::StratumKawPOW);
            break;
        }
        case algo::ALGORITHM::MEOWPOW:
        {
            stratum = NEW(stratum::StratumMeowPOW);
            break;
        }
        case algo::ALGORITHM::FIROPOW:
        {
            stratum = NEW(stratum::StratumFiroPOW);
            break;
        }
        case algo::ALGORITHM::EVRPROGPOW:
        {
            stratum = NEW(stratum::StratumEvrprogPOW);
            break;
        }
        case algo::ALGORITHM::AUTOLYKOS_V2:
        {
            stratum = NEW(stratum::StratumAutolykosV2);
            break;
        }
        case algo::ALGORITHM::BLAKE3:
        {
            stratum = NEW(stratum::StratumBlake3);
            break;
        }
        case algo::ALGORITHM::UNKNOWN:
        {
            break;
        }
    }

    if (nullptr == stratum)
    {
        logErr() << "Fail alloc stratum for " << algorithm;
    }

    stratum->stratumType = config.mining.stratumType;

    switch(stratum->stratumType)
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
