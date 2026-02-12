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


uint32_t common::Cli::getMockerResolverUpdateMemorySleep() const
{
    if (true == contains("tool_mocker_resolver_update_memory_sleep"))
    {
        return params["tool_mocker_resolver_update_memory_sleep"].as<uint32_t>();
    }

    return 1000u;
}

#endif // TOOLS_ENABLE && TOOL_MOCKER
