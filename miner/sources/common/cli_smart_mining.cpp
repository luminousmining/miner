#include <common/cli.hpp>

extern std::vector<std::string> optionSmartMiningWallet;
extern std::vector<std::string> optionSmartMiningPool;


bool common::Cli::isSmartMining() const
{
    if (true == contains("sm_wallet"))
    {
        return true;
    }
    if (true == contains("sm_pool"))
    {
        return true;
    }
    return false;
}


common::Cli::customTupleStrStr common::Cli::getSmartMiningWallet() const
{
    return getCustomParamsStrStr("sm_wallet", optionSmartMiningWallet);
}


common::Cli::customTupleStrStrU32 common::Cli::getSmartMiningPool() const
{
    return getCustomParamsStrStrU32("sm_pool", optionSmartMiningPool);
}
