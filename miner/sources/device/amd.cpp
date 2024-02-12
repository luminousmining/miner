#include <common/log/log.hpp>
#include <device/amd.hpp>
#include <resolver/amd/amd.hpp>


device::DeviceAmd::~DeviceAmd()
{
    cleanUp();
}


bool device::DeviceAmd::initialize()
{
    clContext = cl::Context(clDevice);
    clQueue = cl::CommandQueue(clContext, clDevice);

    resolver::ResolverAmd* const resolverAmd{ dynamic_cast<resolver::ResolverAmd* const>(resolver) };
    if (nullptr == resolverAmd)
    {
        return false;
    }

    resolverAmd->setDevice(&clDevice);
    resolverAmd->setContext(&clContext);
    resolverAmd->setQueue(&clQueue);

    return true;
}


void device::DeviceAmd::cleanUp()
{
}

