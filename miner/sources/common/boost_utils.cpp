#include <common/boost_utils.hpp>


bool common::boostJsonContains(
    boost::json::object const& obj,
    std::string const& keyName)
{
    return obj.find(keyName) != obj.end();
}
