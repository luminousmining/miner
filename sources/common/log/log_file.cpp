#include <common/config.hpp>
#include <common/log/log_file.hpp>


common::LoggerFile& common::LoggerFile::instance()
{
    static common::LoggerFile handler{};
    return handler;
}


bool common::LoggerFile::isOpen() const
{
    return fd.is_open();
}


void common::LoggerFile::openFilename()
{
    common::Config const& config { common::Config::instance() };

    if (false == config.log.file.empty())
    {
        fd.open(config.log.file, std::ios_base::app);
        if (true == isOpen())
        {
            logInfo() << "Logfile[" << config.log.file << "]";
        }
    }
}


void common::LoggerFile::write(
    std::string const& message)
{
    fd.write(message.c_str(), message.size());
    fd.write("\n", 1);
    fd.flush();
}
