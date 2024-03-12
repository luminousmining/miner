#include <common/log/log.hpp>
#include <resolver/nvidia/nvidia.hpp>


size_t resolver::ResolverNvidia::getCurrentIndex() const
{
    return currentIndexStream;
}


size_t resolver::ResolverNvidia::getNextIndex() const
{
    return nextIndexStream;
}


cudaStream_t resolver::ResolverNvidia::getCurrentStream()
{
    return cuStream[currentIndexStream];
}


cudaStream_t resolver::ResolverNvidia::getNextStream()
{
    return cuStream[nextIndexStream];
}


void resolver::ResolverNvidia::swapStream()
{
    currentIndexStream = nextIndexStream;
    nextIndexStream = !nextIndexStream;
}
