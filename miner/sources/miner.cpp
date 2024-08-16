#include <string>

#include <api/api.hpp>
#include <common/app.hpp>
#include <common/config.hpp>
#include <common/log/log.hpp>
#include <device/device_manager.hpp>
#include <network/network.hpp>


static void welcome()
{
    logCustom()
        << common::COLOR_YELLOW << "LuminousMiner v"
        << std::to_string(common::VERSION_MAJOR)
        << "."
        << std::to_string(common::VERSION_MINOR);
}


int main(
    int const argc,
    char** argv)
{
    try
    {
        ////////////////////////////////////////////////////////////////////////
        device::DeviceManager deviceManager{};
        common::Config& config { common::Config::instance() };
        api::ServerAPI serverAPI{};

        ////////////////////////////////////////////////////////////////////////
        welcome();

        ////////////////////////////////////////////////////////////////////////
        if (false == config.load(argc, argv))
        {
            return 1;
        }
        if (true == config.cli.contains("help"))
        {
            config.cli.help();
            return 0;
        }

        ////////////////////////////////////////////////////////////////////////
        serverAPI.setPort(8080);
        if (false == serverAPI.bind())
        {
            return 1;
        }

        ////////////////////////////////////////////////////////////////////////
        if (false == deviceManager.initialize())
        {
            return 1;
        }
        serverAPI.setDeviceManager(&deviceManager);
        if (common::PROFILE::STANDARD == config.profile)
        {
            deviceManager.run();
            deviceManager.connectToPools();
        }
        else
        {
            deviceManager.connectToSmartMining();
        }
    }
    catch(std::exception const& e)
    {
        logErr() << e.what();
        return 1;
    }

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "quitting...";
    return 0;
}
