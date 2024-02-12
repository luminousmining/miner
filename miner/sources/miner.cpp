#include <string>

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

        ////////////////////////////////////////////////////////////////////////
        welcome();

        ////////////////////////////////////////////////////////////////////////
        if (false == config.load(argc, argv))
        {
            logErr() << "Fail load";
            return 1;
        }

        ////////////////////////////////////////////////////////////////////////
        if (false == deviceManager.initialize())
        {
            return 1;
        }
        deviceManager.run();
        deviceManager.connectToPools();
    }
    catch(std::exception const& e)
    {
        logErr() << e.what();
        return 1;
    }

    ////////////////////////////////////////////////////////////////////////////
    return 0;
}
