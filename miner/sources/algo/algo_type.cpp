#include <algo/algo_type.hpp>


std::string algo::toString(algo::ALGORITHM const algorithm)
{
    switch (algorithm)
    {
        case algo::ALGORITHM::SHA256: { return "sha256"; }
        case algo::ALGORITHM::ETHASH: { return "ethash"; }
        case algo::ALGORITHM::ETCHASH: { return "etchash"; }
        case algo::ALGORITHM::PROGPOW: { return "progpow"; }
        case algo::ALGORITHM::KAWPOW: { return "kawpow"; }
        case algo::ALGORITHM::FIROPOW: { return "firopow"; }
        case algo::ALGORITHM::AUTOLYKOS_V2: { return "autolykosv2"; }
        case algo::ALGORITHM::UNKNOW: { return "unknow"; }
    }

    return "";
}
