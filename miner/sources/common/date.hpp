#include <string>
#include <sstream>

#include <boost/date_time.hpp>


namespace common
{
    inline
    std::string getDate()
    {
        auto const& localTime{ boost::posix_time::second_clock::local_time() };
        auto const& date{ localTime.date() };
        auto const& time{ localTime.time_of_day() };

        auto const year{ date.year() };
        auto const month{ date.month() };
        auto const day{ date.day() };

        auto const hours{ time.hours() };
        auto const minutes{ time.minutes() };
        auto const seconds{ time.seconds() };

        std::stringstream ss;
        ss << "("
           << year
           << "/"
           << month
           << "/"
           << (day >= 10 ? "" : "0") << day
           << ")("
           << (hours >= 10 ? "" : "0") << hours
           << ":"
           << (minutes >= 10 ? "" : "0") << minutes
           << ":"
           << (seconds >= 10 ? "" : "0") << seconds
           << ")";

        return ss.str();
    }
}
