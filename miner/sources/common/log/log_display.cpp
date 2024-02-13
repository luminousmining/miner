#if defined(_WIN32)
#include <windows.h>
#include <wincrypt.h>
#endif

#include <iostream>

#include <common/log/log.hpp>
#include <common/log/log_display.hpp>


common::LoggerDisplay common::Logger::logDisplay;


common::LoggerDisplay::LoggerDisplay() noexcept
{
#if defined(_WIN32)
    HANDLE hOut{ GetStdHandle(STD_OUTPUT_HANDLE) };
    if (hOut != INVALID_HANDLE_VALUE)
    {
        DWORD dwMode{ 0 };
        if (GetConsoleMode(hOut, &dwMode))
        {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            if (FALSE == SetConsoleMode(hOut, dwMode))
            {
                logErr() << "Cannot set the console mode color.";
            }
        }
    }
    if (FALSE == SetConsoleOutputCP(CP_UTF8))
    {
        logErr() << "Cannot set the console output type.";
    }
#endif
}


void common::LoggerDisplay::print(common::LogInfo const& info)
{
    try
    {
        switch(info.typeLog)
        {
            case TYPELOG::__CUSTOM:                                       break;
            case TYPELOG::__INFO:    std::cout << common::COLOR_DEFAULT;  break;
            case TYPELOG::__WARNING: std::cout << common::COLOR_GREEN;    break;
            case TYPELOG::__ERROR:   std::cout << common::COLOR_RED;      break;
            case TYPELOG::__TRACE:   std::cout << common::COLOR_MAGENTA;   break;
            case TYPELOG::__DEBUG:   std::cout << common::COLOR_PURPLE;   break;
            default:                 std::cout << common::COLOR_DEFAULT;  break;
        }

        std::cout
           << info.message
           << "\n"
           << COLOR_DEFAULT;
        std::cout.flush();
    }
    catch(...)
    {
        // nothing
    }
}
