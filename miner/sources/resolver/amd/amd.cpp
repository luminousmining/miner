#include <common/cast.hpp>
#include <resolver/amd/amd.hpp>


void resolver::ResolverAmd::setDevice(
    cl::Device* const device)
{
    clDevice = device;
}


void resolver::ResolverAmd::setContext(
    cl::Context* const context)
{
    clContext = context;
}


void resolver::ResolverAmd::setQueue(
    cl::CommandQueue* const queue)
{
    clQueue = queue;
}


uint32_t resolver::ResolverAmd::getMaxGroupSize() const
{
    return castU32(clDevice->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
}
