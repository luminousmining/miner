#pragma once

#include <fstream>

namespace common
{
    struct LoggerFile
    {
    public:
        static LoggerFile& instance();
        bool isOpen() const;
        void openFilename();
        void write(std::string const& message);

    private:
        std::ofstream fd;
    };
}
