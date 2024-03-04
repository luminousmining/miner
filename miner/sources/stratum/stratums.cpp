#include <common/log/log.hpp>
#include <stratum/stratums.hpp>


stratum::Stratum* stratum::NewStratum(
    algo::ALGORITHM const algorithm)
{
    stratum::Stratum* stratum { nullptr };
    switch (algorithm)
    {
        case algo::ALGORITHM::SHA256:
        {
            stratum = new (std::nothrow) stratum::StratumSha256;
            break;
        }
        case algo::ALGORITHM::ETHASH:
        {
            stratum = new (std::nothrow) stratum::StratumEthash;
            break;
        }
        case algo::ALGORITHM::ETCHASH:
        {
            stratum = new (std::nothrow) stratum::StratumEtchash;
            break;
        }
        case algo::ALGORITHM::PROGPOW:
        {
            stratum = new (std::nothrow) stratum::StratumProgPOW;
            break;
        }
        case algo::ALGORITHM::KAWPOW:
        {
            stratum = new (std::nothrow) stratum::StratumKawPOW;
            break;
        }
        case algo::ALGORITHM::FIROPOW:
        {
            stratum = new (std::nothrow) stratum::StratumFiroPOW;
            break;
        }
        case algo::ALGORITHM::EVRPROGPOW:
        {
            stratum = new (std::nothrow) stratum::StratumEvrprogPOW;
            break;
        }
        case algo::ALGORITHM::AUTOLYKOS_V2:
        {
            stratum = new (std::nothrow) stratum::StratumAutolykosV2;
            break;
        }
        case algo::ALGORITHM::UNKNOW:
        {
            break;
        }
    }

    if (nullptr == stratum)
    {
        logErr() << "Fail alloc stratum for " << algorithm;
    }

    return stratum;
}
