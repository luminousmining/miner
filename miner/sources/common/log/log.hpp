#pragma once

#include <sstream>

#include <algo/hash.hpp>
#include <algo/algo_type.hpp>
#include <common/log/log_display.hpp>
#include <common/log/log_type.hpp>
#include <stratum/job_info.hpp>


namespace common
{
    void setLogLevel(common::TYPELOG const typeLog);

    struct Logger
    {
    public:
        static common::LoggerDisplay logDisplay;

        Logger() = delete;
        Logger(char const* function, size_t line, common::TYPELOG tl);
        ~Logger() noexcept;

        Logger& operator<<(bool const value);
        Logger& operator<<(algo::ALGORITHM const algo);
        Logger& operator<<(algo::hash256 const& hash);
        Logger& operator<<(algo::hash512 const& hash);
        Logger& operator<<(algo::hash1024 const& hash);
        Logger& operator<<(algo::hash2048 const& hash);
        Logger& operator<<(std::string const& str);
        Logger& operator<<(stratum::StratumJobInfo const& jobInfo);

        template<typename T>
        inline
        Logger& operator<<(T const& t)
        {
            ss << t;
            return *this;
        }

    protected:
        std::stringstream ss;
        common::TYPELOG   typeLog{ common::TYPELOG::__INFO };
    };
}


#define logCustom() common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__CUSTOM)
#define logErr()    common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__ERROR)
#define logInfo()   common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__INFO)
#define logWarn()   common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__WARNING)
#define logTrace()  common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__TRACE)
#define logDebug()  common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__DEBUG)

#if defined(_DEBUG)
#define __TRACE() { logTrace(); }
#else
#define __TRACE() { logTrace() << __FUNCTION__ << ":" << __LINE__; }
#endif
