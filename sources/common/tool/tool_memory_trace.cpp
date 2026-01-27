#if defined (TOOL_TRACE_MEMORY)

#include <common/log/log.hpp>
#include <common/tool/tool_memory_trace.hpp>


common::tool::ToolMemoryTrace& common::tool::ToolMemoryTrace::instance()
{
    static common::tool::ToolMemoryTrace handler{};
    return handler;
}


void common::tool::ToolMemoryTrace::registerAllocation(
    void* ptr,
    size_t const size,
    char const* functionName,
    uint32_t const line)
{
    common::tool::ToolMemoryTrace::AllocationInfo info{};
    info.functionName = functionName;
    info.line = line;
    info.size = size;

    allocations[ptr] = info;
    ++totalAllocated;
    ++currentAllocated;

    logInfo()
        << "[" << functionName << "][" << line << "] "
        << "alloc " << size << " bytes "
        << "at address 0x"<< std::hex << ptr;
}


void common::tool::ToolMemoryTrace::registerDesallocation(
    void* ptr,
    char const* functionName,
    uint32_t const line)
{
    auto it = allocations.find(ptr);
    if (it != allocations.end())
    {
        allocations.erase(it);
        ++totalFreed;
        --currentAllocated;
        logInfo()
            << "[" << functionName << "][" << line << "] "
            << "free at address 0x"<< std::hex << ptr;
    }
}


void common::tool::ToolMemoryTrace::show() const
{
    std::stringstream ss;
    ss
        << "=== Trace Memory Manager ===" << "\n"
        << "Total allocated: " << totalAllocated << "\n"
        << "Total freed: " << totalFreed << "\n"
        << "Currently allocated: " << currentAllocated
        ;

    if (0 != currentAllocated)
    {
        ss << "\n" <<"Non-freed allocations:";
        for (auto const& pair : allocations)
        {
           ss
            << "  Ptr: " << pair.first 
            << " | Function: " << pair.second.functionName 
            << " | Line: " << pair.second.line 
            << " | Size: " << pair.second.size << " bytes" << std::endl;
        }
    }

    logInfo() << ss.str();
}

#endif
