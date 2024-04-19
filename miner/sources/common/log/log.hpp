#pragma once

#include <sstream>
#include <string>
#include <string_view>

#include <boost/beast/core/string_type.hpp>

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
        Logger& operator<<(algo::hash3072 const& hash);
        Logger& operator<<(algo::hash4096 const& hash);
        Logger& operator<<(std::string const& str);
        Logger& operator<<(std::string_view const& str);
        Logger& operator<<(boost::beast::string_view const& str);
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

#define deviceCustom() common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__CUSTOM)  << "Device[" << id << "]: "
#define deviceErr()    common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__ERROR)   << "Device[" << id << "]: "
#define deviceInfo()   common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__INFO)    << "Device[" << id << "]: "
#define deviceWarn()   common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__WARNING) << "Device[" << id << "]: "
#define deviceTrace()  common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__TRACE)   << "Device[" << id << "]: "
#define deviceDebug()  common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__DEBUG)   << "Device[" << id << "]: "

#define resolverCustom() common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__CUSTOM)  << "Device[" << deviceId << "]: "
#define resolverErr()    common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__ERROR)   << "Device[" << deviceId << "]: "
#define resolverInfo()   common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__INFO)    << "Device[" << deviceId << "]: "
#define resolverWarn()   common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__WARNING) << "Device[" << deviceId << "]: "
#define resolverTrace()  common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__TRACE)   << "Device[" << deviceId << "]: "
#define resolverDebug()  common::Logger(__FUNCTION__, __LINE__, common::TYPELOG::__DEBUG)   << "Device[" << deviceId << "]: "

#define __TRACE() { logTrace() << __FUNCTION__ << ":" << __LINE__; }
#define __TRACE_DEVICE()                                                       \
    {                                                                          \
        logTrace() << __FUNCTION__ << ":" << __LINE__                          \
        << ": device[" << id << "]";                                           \
    }
