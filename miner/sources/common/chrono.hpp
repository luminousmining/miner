#pragma once


#include <chrono>


namespace common
{
    enum class CHRONO_UNIT : uint8_t
    {
        SEC,
        MS,
        US,
        NS
    };

    class ChronoGuard
    {
    public:

        ChronoGuard(common::CHRONO_UNIT unit);
        ~ChronoGuard();

    private:
        common::CHRONO_UNIT unit{ common::CHRONO_UNIT::SEC };
        std::chrono::system_clock::time_point tmStart{};
    };


    class Chrono
    {
    public:
        Chrono() = default;
        ~Chrono() = default;

        void start();
        void stop();
        uint64_t elapsed(common::CHRONO_UNIT unit) const;

    private:
        std::chrono::system_clock::time_point tmStart{};
        std::chrono::system_clock::time_point tmEnd{};
    };
}
