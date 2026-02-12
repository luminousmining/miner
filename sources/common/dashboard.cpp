#include <sstream>

#include <common/cast.hpp>
#include <common/dashboard.hpp>
#include <common/log/log.hpp>


void common::Dashboard::setDate(std::string const& newDateInfo)
{
    dateInfo.assign(newDateInfo);
}


void common::Dashboard::setTag(std::string const& newTag)
{
    tag.clear();
    tag += "[";
    tag += newTag;
    tag += "]";
}


void common::Dashboard::setTitle(std::string const& newTitle)
{
    Title t{ newTitle, newTitle.size() };
    title = t;
}


void common::Dashboard::setFooter(
    std::string const& key,
    std::string const& value)
{
    footer.clear();
    footer += key;
    footer += ": ";
    footer += value;
}


void common::Dashboard::addColumn(std::string const& name)
{
    common::Dashboard::Column col{ name, name.size() };
    columns.emplace_back(col);
}


void common::Dashboard::addLine(std::vector<std::string> const& line)
{
    std::vector<common::Dashboard::Line> data;
    for (auto const& value : line)
    {
        common::Dashboard::Line l{ value, value.size() };
        data.emplace_back(std::move(l));
    }
    lines.emplace_back(data);
}


void common::Dashboard::resetLines()
{
    for (auto& line : lines)
    {
        line.clear();
    }
    lines.clear();

    for (auto& col : columns)
    {
        col.size = col.name.size();
    }
}


void common::Dashboard::compute()
{
    totalSize = 0u;
    for (auto i{ 0ul }; i < columns.size(); ++i)
    {
        auto& col{ columns.at(i) };
        for (auto const& line : lines)
        {
            auto const& value{ line.at(i) };
            if (col.size < value.size)
            {
                col.size = value.size;
            }
        }
        col.size += 1u;
        totalSize += castU32(col.size + 2u);
    }
}


void common::Dashboard::addPrefix(std::stringstream& ss) const
{
    if (false == tag.empty())
    {
        ss << dateInfo << " - " << tag << " - ";
    }
    else
    {
        ss << dateInfo << " - ";
    }
}


void common::Dashboard::show()
{
    compute();

    std::stringstream ss;

    // Draw the title
    addPrefix(ss);
    uint32_t const titleGap{ (totalSize - cast32(title.size) - 2u) / 2u };
    for (uint32_t i{ 0u }; i < titleGap; ++i)
    {
        ss << "=";
    }
    ss << " " << title.name << " ";
    for (uint32_t i{ 0u }; i < titleGap + 1; ++i)
    {
        ss << "=";
    }
    ss << "\n";

    // Draw the columns
    addPrefix(ss);
    for (auto const& col : columns)
    {
        ss << "| " << col.name;
        for (auto i{ col.name.size() }; i < col.size; ++i)
        {
            ss << " ";
        }
    }
    ss << "|";
    ss << "\n";

    // Draw the lines
    for (auto const& line : lines)
    {
        addPrefix(ss);
        for (auto i{ 0ul }; i < columns.size(); ++i)
        {
            auto const& col{ columns.at(i) };
            auto const& value{ line.at(i) };
            ss << "| ";
            ss << value.value;
            for (auto columGap{ value.size }; columGap < col.size; ++columGap)
            {
                ss << " ";
            }
        }
        ss << "|";
        ss << "\n";
    }

    // Draw the last line
    addPrefix(ss);
    auto const width{ totalSize + 1};
    for (auto i{ 0ul }; i < width; ++i)
    {
        ss << "=";
    }

    // Draw footer
    if (false == footer.empty())
    {
        ss << "\n";
        addPrefix(ss);
        ss << "| " << footer;
        auto footerGap{ totalSize - footer.size() - 2 };
        for (auto i{ 0ul }; i < footerGap; ++i)
        {
            ss << " ";
        }
        ss << "|";
        ss << "\n";
        addPrefix(ss);
        for (auto i{ 0ul }; i < width; ++i)
        {
            ss << "=";
        }
    }
    logCustom() << ss.str();
}
