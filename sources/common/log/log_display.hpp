#pragma once

#include <common/log/log_type.hpp>


namespace common
{
    struct LoggerDisplay
    {
        LoggerDisplay() noexcept;
        ~LoggerDisplay() noexcept = default;
        void print(common::LogInfo const& info);
    };
}
