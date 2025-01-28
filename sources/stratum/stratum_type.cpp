#include <stratum/stratum_type.hpp>


std::string stratum::toString(stratum::STRATUM_TYPE const stratumType)
{
    using namespace std::string_literals;

    switch(stratumType)
    {
        case stratum::STRATUM_TYPE::ETHEREUM_V1:
        {
            return "Ethereum/1.0.0"s;
        }
        case stratum::STRATUM_TYPE::ETHEREUM_V2:
        {
            return "Ethereum/2.0.0"s;
        }
        case stratum::STRATUM_TYPE::ETHPROXY:
        {
            return "Ethereum/Proxy"s;
        }
    }
    return "";
}
