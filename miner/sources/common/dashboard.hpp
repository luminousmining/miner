#pragma once

#include <string>
#include <vector>


namespace common
{
    class Dashboard
    {
    public:
        Dashboard() = default;
        ~Dashboard() = default;

        void setDate(std::string newDateInfo);
        void setTag(std::string newTag);
        void setTitle(std::string const& newTitle);
        void setFooter(std::string const& key,
                       std::string const& value);
        void addColumn(std::string const& name);
        void addLine(std::vector<std::string> const& line);
        void resetLines();
        void compute();
        void show();

    private:
        struct Column
        {
            std::string name;
            size_t size{ 0u };
        };

        struct Line
        {
            std::string value;
            size_t size{ 0u };
        };

        struct Title
        {
            std::string name{};
            size_t size{ 0u };
        };

        Title title{};
        uint32_t totalSize{ 0u };
        std::string dateInfo{};
        std::string tag{};
        std::string footer{};
        std::vector<Column> columns{};
        std::vector<std::vector<Line>> lines;

        void addPrefix(std::stringstream& ss) const;
    };
}
