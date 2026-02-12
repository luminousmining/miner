#if defined(TOOLS_ENABLE) &&  defined(TOOL_MOCKER)

#include <common/cli/cli.hpp>


uint32_t common::Cli::getMockerResolverCount() const
{
    if (true == contains("tool_mocker_resolver_count"))
    {
        return params["tool_mocker_resolver_count"].as<uint32_t>();
    }

    return 8u;
}


#endif // TOOLS_ENABLE && TOOL_MOCKER
