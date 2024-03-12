#include <common/cli.hpp>


extern std::vector<std::string> devOptionDoubleStream;


common::Cli::customTupleU32 common::Cli::getDevDeviceDoubleStream() const
{
    return getCustomParamsU32("dev_device_2stream", devOptionDoubleStream);
}
