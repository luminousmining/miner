#if defined(TOOL_MOCKER)

#include <common/log/log.hpp>
#include <device/mocker.hpp>


bool device::DeviceMocker::initialize()
{
    logInfo() << "Initialize device mocker";
    return true;
}


void device::DeviceMocker::cleanUp()
{
    logInfo() << "Clean up device mocker";
}

#endif
