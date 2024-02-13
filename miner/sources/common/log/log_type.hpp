#pragma once

#include <string>


namespace common
{
    constexpr char const* COLOR_RESET   { "\x1B[0m"  };
    constexpr char const* COLOR_RED     { "\x1B[31m" };
    constexpr char const* COLOR_GREEN   { "\x1B[32m" };
    constexpr char const* COLOR_YELLOW  { "\x1B[33m" };
    constexpr char const* COLOR_DEFAULT { "\x1B[37m" };
    constexpr char const* COLOR_PURPLE  { "\x1B[95m" };
    constexpr char const* COLOR_MAGENTA { "\x1B[35m" };

    enum class TYPELOG : uint8_t
    {
        __CUSTOM,
        __ERROR,
        __INFO,
        __WARNING,
        __TRACE,
        __DEBUG,
    };

    struct LogInfo
    {
        common::TYPELOG typeLog{ TYPELOG::__INFO };
        std::string     message;
    };
}
