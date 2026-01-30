#pragma once


#include <chrono>


namespace common
{
    constexpr uint32_t SEC_TO_MS{ 1000u };
    constexpr uint32_t SEC_TO_US{ 1000000u };
    constexpr uint32_t SEC_TO_NS{ 1000000000u };

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
        ChronoGuard(std::string const& text, common::CHRONO_UNIT unit);
        ~ChronoGuard();

    private:
        std::string text{};
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
